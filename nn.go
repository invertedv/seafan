package seafan

// NN functionality

import (
	"encoding/json"
	"fmt"
	"gorgonia.org/golgi"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// CostFunc function proto for cost functions
type CostFunc func(model NNet) *G.Node

// NNet interface for NN models
type NNet interface {
	Inputs() G.Nodes            // input nodes
	Features() G.Nodes          // predictors
	Fitted() G.Result           // model output
	Params() G.Nodes            // model weights
	Obs() *G.Node               // observed values
	CostFn() CostFunc           // cost function of fitting
	Cost() *G.Node              // cost node in graph
	Fwd()                       // forward pass
	G() *G.ExprGraph            // return graph
	Save(fileRoot string) error // save model
}

// NNModel structure
type NNModel struct {
	name      string       // name of model
	g         *G.ExprGraph // model graph
	paramsW   G.Nodes      // weight parameters
	paramsB   G.Nodes      // bias parameters
	paremsEmb G.Nodes      // embedding parameters
	output    G.Result     // graph output
	inputsC   G.Nodes      // continuous (including one-hot) inputs
	inputsE   G.Nodes      // embedding inputs
	obs       *G.Node      // observed values for model fit
	cost      *G.Node      // cost node for model build
	construct FTypes       // output, input FType
	drops     Drops        // drop out layers
	costFn    CostFunc     // costFn corresponding to cost *G.Node
}

// Name returns model name
func (m *NNModel) Name() string {
	return m.name
}

func (m *NNModel) String() string {
	if m.construct == nil {
		return "No model"
	}
	str := fmt.Sprintf("%s\nInputs:\n", m.name)
	for ind := 1; ind < len(m.construct); ind++ {
		fld := "\t" + strings.ReplaceAll(m.construct[ind].String(), "\t", "\t\t")
		str = fmt.Sprintf("%s%s", str, fld)
	}
	fld := "\t" + strings.ReplaceAll(m.construct[0].String(), "\t", "\t\t")
	str = fmt.Sprintf("%s\nTarget:\n%s", str, fld)
	if m.cost != nil {
		str = fmt.Sprintf("%s\nCost function: %s\n", str, m.cost.Name())
	}
	bSize := m.inputsC[0].Shape()[0]
	str = fmt.Sprintf("%sBatch size: %d\n", str, bSize)
	str = fmt.Sprintf("%sNN structure:\n", str)
	for ind, n := range m.paramsW {
		if d := m.drops.Get(ind); d != nil {
			str = fmt.Sprintf("%s\tDrop Layer (probability = %0.2f)\n", str, d.DropProb)
		}
		addon := ""
		if ind == len(m.paramsW)-1 {
			addon = " (output)"
		}
		str = fmt.Sprintf("%s\tFC Layer %d: %v%s\n", str, ind, n.Shape(), addon)
	}

	return str
}

// CostFn returns cost function
func (m *NNModel) CostFn() CostFunc {
	return m.costFn
}

// Cost returns cost node
func (m *NNModel) Cost() *G.Node {
	return m.cost
}

// FitSlice returns fitted values as a slice
func (m *NNModel) FitSlice() []float64 {
	return m.output.Nodes()[0].Value().Data().([]float64)
}

// ObsSlice returns target values as a slice
func (m *NNModel) ObsSlice() []float64 {
	return m.obs.Value().Data().([]float64)
}

// CostFlt returns the value of the cost node
func (m *NNModel) CostFlt() float64 {
	return m.cost.Value().Data().(float64)
}

// Obs returns the target value as a node
func (m *NNModel) Obs() *G.Node {
	return m.obs
}

// Fitted returns fitted values as a G.Result
func (m *NNModel) Fitted() G.Result {
	return m.output
}

// Inputs returns input (continuous+embedded+observed) inputs
func (m *NNModel) Inputs() G.Nodes {
	n := append(m.inputsC, m.inputsE...)
	return append(n, m.obs)
}

// Features returns the model input features (continuous+embedded)
func (m *NNModel) Features() G.Nodes {
	return append(m.inputsC, m.inputsE...)
}

// Params retursn the model parameter nodes (weights, biases, embeddings)
func (m *NNModel) Params() G.Nodes {
	p := append(m.paramsW, m.paramsB...)
	p = append(p, m.paremsEmb...)
	return p
}

// G returns model graph
func (m *NNModel) G() *G.ExprGraph {
	return m.g
}

// Drop specifies a dropout layer.  It occurs in the graph after dense layer AfterLayer (the input layer is layer 0).
type Drop struct {
	AfterLayer int     // insert dropout after layer AfterLayer
	DropProb   float64 // dropout probability
}

type Drops []*Drop

// Get returns the dropout layer that occurs after dense layer after
func (d Drops) Get(after int) *Drop {
	for _, l := range d {
		if l.AfterLayer == after {
			return l
		}
	}
	return nil
}

// NNOpts -- NNModel options
type NNOpts func(model1 *NNModel)

// WithDropOuts adds dropout layers
func WithDropOuts(drops Drops) NNOpts {
	f := func(m *NNModel) {
		m.drops = drops
		m.Fwd()
	}
	return f
}

// WithCostFn adds a cost function
func WithCostFn(cf CostFunc) NNOpts {
	f := func(m *NNModel) {
		m.costFn = cf
		m.cost = cf(m)
	}
	return f
}

func WithName(name string) NNOpts {
	f := func(m *NNModel) {
		m.name = name
	}
	return f
}

// NewNNModel creates a new NN model. The modSpec input can be created by ByFormula.
func NewNNModel(bSize int, modSpec []*FType, hidden []int, no ...NNOpts) *NNModel {
	g := G.NewGraph()
	xs := make(G.Nodes, 0)
	embParm := make(G.Nodes, 0) // embedding parameters
	xEmInp := make(G.Nodes, 0)  // one-hot input
	xEmProd := make(G.Nodes, 0) // product of one-hot input and embedding parameters
	// work through the features
	for ind, f := range modSpec {
		// first element is the target--skip
		if ind == 0 {
			continue
		}
		switch f.Role {
		case FRCts:
			x := G.NewTensor(g, tensor.Float64, 2, G.WithName(f.Name), G.WithShape(bSize, 1))
			xs = append(xs, x)
		case FROneHot:
			x := G.NewTensor(g, tensor.Float64, 2, G.WithName(f.Name), G.WithShape(bSize, f.Cats))
			xs = append(xs, x)
		case FREmbed:
			xemb := G.NewTensor(g, tensor.Float64, 2, G.WithName(f.Name), G.WithShape(bSize, f.Cats))
			xEmInp = append(xEmInp, xemb)
			wemb := G.NewTensor(g, G.Float64, 2, G.WithName(f.Name+"Embed"), G.WithShape(f.Cats, f.EmbCols), G.WithInit(G.GlorotU(1)))
			embParm = append(embParm, wemb)
			z := G.Must(G.Mul(xemb, wemb))
			xEmProd = append(xEmProd, z)
		}
	}

	// inputs
	xall := G.Must(G.Concat(1, xs...))

	// add inputs to embeddings, if present
	if len(xEmInp) > 0 {
		zemb := G.Must(G.Concat(1, xEmProd...)) // embeddings for input to FC layer
		xall = G.Must(G.Concat(1, xall, zemb))
	}

	// target
	obsF := modSpec[0]
	var yoh *G.Node
	// obsF.Cats = 0 if the target is continuous
	switch {
	case obsF.Cats <= 1:
		yoh = G.NewTensor(g, tensor.Float64, 2, G.WithName(obsF.Name), G.WithShape(bSize, 1))
	default:
		yoh = G.NewTensor(g, tensor.Float64, 2, G.WithName(obsF.Name), G.WithShape(bSize, obsF.Cats))
	}

	outLRows := xall.Shape()[1] // layer output dim
	parW := make(G.Nodes, 0)
	parB := make(G.Nodes, 0)
	if hidden != nil {
		dims := []int{outLRows}
		dims = append(dims, hidden...)
		outLRows = hidden[len(hidden)-1]
		for ind := 1; ind < len(dims); ind++ {
			nmw := "lWeights" + strconv.Itoa(ind)
			nmb := "lBias" + strconv.Itoa(ind)
			w := G.NewTensor(g, tensor.Float64, 2, G.WithName(nmw), G.WithShape(dims[ind-1], dims[ind]), G.WithInit(G.GlorotN(1.0)))
			b := G.NewTensor(g, tensor.Float64, 2, G.WithName(nmb), G.WithShape(1, dims[ind]), G.WithInit(G.GlorotN(1.0)))
			parW = append(parW, w)
			parB = append(parB, b)
		}
	}
	// if target categorical, we drop the last category.  This eliminates unidentifiability issues
	sub := 1
	if yoh.Shape()[1] == 1 {
		sub = 0
	}
	// weights & bias for output layer
	lw := G.NewTensor(g, tensor.Float64, 2, G.WithName("lWeightsOut"), G.WithShape(outLRows, yoh.Shape()[1]-sub), G.WithInit(G.GlorotN(1.0)))
	lb := G.NewTensor(g, tensor.Float64, 2, G.WithName("lBiasOut"), G.WithShape(1, yoh.Shape()[1]-sub), G.WithInit(G.GlorotN(1.0)))
	parW = append(parW, lw)
	parB = append(parB, lb)

	nn := &NNModel{
		g:         g,
		paramsW:   parW,
		paramsB:   parB,
		paremsEmb: embParm,
		inputsC:   xs,
		inputsE:   xEmInp,
		obs:       yoh,
		construct: modSpec,
		drops:     nil,
	}
	nn.Fwd() // init forward pass
	// add user opts
	for _, o := range no {
		o(nn)
	}
	return nn
}

// Fwd builds forward pass
func (m *NNModel) Fwd() {
	// input nodes
	xall := G.Must(G.Concat(1, m.inputsC...))
	// add embeddings
	if len(m.inputsE) > 0 {
		zp := make(G.Nodes, 0)
		for ind, x := range m.inputsE {
			z := G.Must(G.Mul(x, m.paremsEmb[ind]))
			zp = append(zp, z)
		}
		emb := G.Must(G.Concat(1, zp...))
		xall = G.Must(G.Concat(1, xall, emb))
	}
	// first layer
	p := m.paramsW[0]
	// add dropout
	if d := m.drops.Get(0); d != nil {
		p = G.Must(G.Dropout(p, d.DropProb))
	}
	// end of first layer
	out := G.Must(G.Mul(xall, p))
	out = G.Must(G.BroadcastAdd(out, m.paramsB[0], nil, []byte{0}))
	// work through hidden layers
	for ind := 1; ind < len(m.paramsW); ind++ {
		p := m.paramsW[ind]
		if d := m.drops.Get(ind); d != nil {
			p = G.Must(G.Dropout(p, d.DropProb))
		}
		out = G.Must(G.Mul(out, p))
		out = G.Must(G.BroadcastAdd(out, m.paramsB[ind], nil, []byte{0}))

	}
	// is output categorical?  Softmax...
	if m.Obs().Shape()[1] > 1 {
		exp := G.Must(G.Exp(out))
		sexp := G.Must(G.Sum(exp, 1))
		sexp = G.Must(G.Add(sexp, G.NewConstant(1.0)))
		phat := G.Must(G.BroadcastHadamardDiv(exp, sexp, nil, []byte{1}))
		phats := G.Must(G.Sum(phat, 1))
		phat1 := G.Must(G.Sub(G.NewConstant(1.0), phats))
		r := phat1.Shape()[0]
		phat1a := G.Must(G.Reshape(phat1, tensor.Shape{r, 1}))
		phat2 := G.Must(G.Concat(1, phat, phat1a))
		m.output = phat2
		return
	}
	m.output = out

}

// struct to save nodes to json file
type saveNode struct {
	Name  string    `json:"name"`
	Dims  []int     `json:"dims"`
	Parms []float64 `json:"parms"`
}

// Save saves a model to disk.  Two files are created: *S.nn stores the structure of the model and
// *P.nn stores the parameters.  Note: dropout layers are not saved.
func (m *NNModel) Save(fileRoot string) (err error) {
	err = nil
	fileP := fileRoot + "P.nn"
	f, err := os.Create(fileP)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()
	ps := make([]saveNode, 0)
	for ind := 0; ind < len(m.Params()); ind++ {
		n := m.Params()[ind]
		p := saveNode{
			Name:  n.Name(),
			Dims:  n.Shape(),
			Parms: n.Value().Data().([]float64),
		}
		ps = append(ps, p)
	}
	jp, err := json.MarshalIndent(ps, "", "  ")
	if err != nil {
		return
	}
	if _, err = f.WriteString(string(jp)); err != nil {
		return
	}

	fileS := fileRoot + "S.nn"
	if err = m.construct.Save(fileS); err != nil {
		return
	}
	return nil
}

// LoadNN restores a previously saved NNModel
func LoadNN(fileRoot string, bSize int) (nn *NNModel, err error) {
	err = nil
	nn = nil
	fileS := fileRoot + "S.nn"
	modSpec, err := LoadFTypes(fileS)
	if err != nil {
		return
	}

	//	f.Close()
	fileP := fileRoot + "P.nn"
	f, err := os.Open(fileP)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()
	js, err := io.ReadAll(f)
	if err != nil {
		return
	}

	data := make([]saveNode, 0)
	if e := json.Unmarshal(js, &data); e != nil {
		return nil, e
	}
	var hidden []int
	for _, n := range data {
		if strings.Contains(n.Name, "Weights") && n.Name != "lWeightsOut" {
			hidden = append(hidden, n.Dims[1])
		}
	}

	nn = NewNNModel(bSize, modSpec, hidden)
	if len(data) != len(nn.Params()) {
		return nil, fmt.Errorf("node count differs")
	}
	for _, d := range data {
		nd := nn.g.ByName(d.Name)[0]
		if nd == nil {
			return nil, fmt.Errorf("node %s not found", d.Name)
		}
		shp := nd.Shape()
		for ind, dim := range shp {
			if dim != d.Dims[ind] {
				return nil, fmt.Errorf("dimensions do not match")

			}
		}
		t := tensor.New(tensor.WithBacking(d.Parms), tensor.WithShape(shp...))
		if err = G.Let(nd, t); err != nil {
			nn = nil
			return
		}
	}
	return
}

func SoftRMS(model NNet) (cost *G.Node) {
	nCats := model.Obs().Shape()[1]
	for ind := 1; ind < nCats; ind++ {
		back := make([]float64, nCats)
		back[ind] = 1.0
		zo := tensor.New(tensor.WithBacking(back), tensor.WithShape(1, nCats))
		mzo := G.NewTensor(model.G(), G.Float64, 2, G.WithName("zo"+strconv.Itoa(ind)), G.WithShape(1, nCats), G.WithValue(zo))
		a := G.Must(G.BroadcastHadamardProd(model.Fitted().Nodes()[0], mzo, nil, []byte{0}))
		b := G.Must(G.BroadcastHadamardProd(model.Obs(), mzo, nil, []byte{0}))
		if ind == 1 {
			cost = G.Must(golgi.RMS(a, b))
		} else {
			cost = G.Must(G.Add(cost, G.Must(golgi.RMS(a, b))))
		}
	}
	return
}

// CrossEntropy cost function
func CrossEntropy(model NNet) (cost *G.Node) {
	cost = G.Must(G.Neg(G.Must(G.Mean(G.Must(G.HadamardProd(G.Must(G.Log(model.Fitted().Nodes()[0])), model.Obs()))))))
	G.WithName("CrossEntropy")(cost)
	return
}

// RMS cost function
func RMS(model NNet) (cost *G.Node) {
	cost = G.Must(golgi.RMS(model.Fitted().Nodes()[0], model.Obs()))
	G.WithName("RMS")(cost)
	return
}

// Fit struct for fitting a NNModel
type Fit struct {
	nn        NNet
	p         Pipeline
	epochs    int
	lrStart   float64
	lrEnd     float64
	outFile   string
	tmpFile   string
	pVal      Pipeline
	inCosts   *XY
	outCosts  *XY
	wait      int
	bestEpoch int
	l2Penalty float64
}

// FitOpts functions add options
type FitOpts func(*Fit)

// NewFit creates a new *Fit.
func NewFit(nn NNet, epochs int, p Pipeline, opts ...FitOpts) *Fit {
	rand.Seed(time.Now().UnixMicro())
	outFile := fmt.Sprintf("%s/NN%d", os.TempDir(), int(rand.Uint32()))
	tmpFile := fmt.Sprintf("%s/NN%d", os.TempDir(), int(rand.Uint32()))
	fit := &Fit{
		nn:      nn,
		epochs:  epochs,
		p:       p,
		outFile: outFile,
		tmpFile: tmpFile,
	}
	for _, o := range opts {
		o(fit)
	}
	return fit
}

// WithL2Reg adds L2 regularization
func WithL2Reg(penalty float64) FitOpts {
	f := func(ft *Fit) {
		ft.l2Penalty = penalty
	}
	return f
}

// WithLearnRate sets a learning rate function that declines linearly across the epochs.
func WithLearnRate(lrStart float64, lrEnd float64) FitOpts {
	f := func(ft *Fit) {
		ft.lrStart = lrStart
		ft.lrEnd = lrEnd
	}
	return f
}

// WithValidation adds a validation Pipeline for early stopping.  The fit is stopped when the validation cost
// does not improve for wait epochs.
func WithValidation(p Pipeline, wait int) FitOpts {
	f := func(ft *Fit) {
		ft.pVal = p
		ft.wait = wait
	}
	return f
}

// WithOutFile specifies the file root name to save the best model.
func WithOutFile(fileName string) FitOpts {
	f := func(ft *Fit) {
		ft.outFile = fileName
	}
	return f
}

// OutFile returns the output file name
func (ft *Fit) OutFile() string {
	return ft.outFile
}

// BestEpoch returns the epoch of the best cost (validation or in-sample--whichever is specified)
func (ft *Fit) BestEpoch() int {
	return ft.bestEpoch
}

// InCosts returns XY: X=epoch, Y=In-sample cost
func (ft *Fit) InCosts() *XY {
	return ft.inCosts
}

// OutCosts returns XY: X=epoch, Y=validation cost
func (ft *Fit) OutCosts() *XY {
	return ft.outCosts
}

// Do is the fitting loop for NNModels.
func (ft *Fit) Do() (err error) {
	//	if ft.nn.Cost == nil {
	//		return fmt.Errorf("no cost node in NNModel")
	//	}
	best := math.MaxFloat64
	ft.bestEpoch = 0
	if _, e := G.Grad(ft.nn.Cost(), ft.nn.Params()...); e != nil {
		log.Fatalln(e)
	}
	vm := G.NewTapeMachine(ft.nn.G(), G.BindDualValues(ft.nn.Params()...))
	defer func() { err = vm.Close() }()

	t := time.Now()
	itv := make([]float64, 0)
	solv := G.NewAdamSolver()
	if ft.l2Penalty > 0.0 {
		G.WithL2Reg(ft.l2Penalty)(solv)
	}
	cv := make([]float64, 0)
	cVal := make([]float64, 0)
	for ep := 1; ep <= ft.epochs; ep++ {
		// check for user specified learning rate
		if ft.lrStart > 0.0 {
			lr := ft.lrEnd + (ft.lrStart-ft.lrEnd)*(1.0-float64(ep)/float64(ft.epochs))
			G.WithLearnRate(lr)(solv)
		}
		// run through batches in one epoch
		for ft.p.Batch(ft.nn.Inputs()) {
			if err = vm.RunAll(); err != nil {
				return
			}
			if err = solv.Step(G.NodesToValueGrads(ft.nn.Params())); err != nil {
				return
			}
			vm.Reset()
		}
		// increment epoch counter in pipeline
		ft.p.Epoch(ft.p.Epoch(-1) + 1)
		itv = append(itv, float64(ep))
		cv = append(cv, ft.nn.Cost().Value().Data().(float64))
		switch ft.pVal == nil {
		case true:
			// judge best epoch by in-sample cost
			if cv[len(cv)-1] < best {
				best = cv[len(cv)-1]
				ft.bestEpoch = ep
				if err = ft.nn.Save(ft.outFile); err != nil {
					return
				}
			}
		case false:
			// find validation cost..save model and load to new graph
			if err = ft.nn.Save(ft.tmpFile); err != nil {
				return err
			}

			var valMod *NNModel
			valMod, err = PredictNN(ft.tmpFile, ft.pVal.BatchSize(), ft.pVal, WithCostFn(ft.nn.CostFn()))
			if err != nil {
				return
			}

			cVal = append(cVal, valMod.CostFlt())
			// judge best epoch by validation cost
			if cVal[len(cVal)-1] < best {
				best = cVal[len(cVal)-1]
				ft.bestEpoch = ep
				if err = ft.nn.Save(ft.outFile); err != nil {
					return
				}
			}
			// check for early stopping
			if ft.wait > 0 && ep > ft.wait {
				checkVal := cVal[len(cVal)-1-ft.wait]
				minC := math.MaxFloat64
				for ind := len(cVal) - ft.wait; ind < len(cVal); ind++ {
					if cVal[ind] < minC {
						minC = cVal[ind]
					}
				}
				if minC > checkVal {
					break
				}
			}
		}
	}
	elapsed := time.Since(t).Minutes()
	if Verbose {
		fmt.Println("best epoch: ", ft.bestEpoch)
		fmt.Printf("elapsed time %0.1f minutes\n", elapsed)
	}
	ft.inCosts, err = NewXY(itv, cv)
	ft.outCosts, err = NewXY(itv, cVal)
	// clean up
	_ = os.Remove(ft.tmpFile + "P.nn")
	_ = os.Remove(ft.tmpFile + "S.nn")
	return
}

// PredictNN reads in a NNModel from disk and populates it with a batch from p.
// Methods such as FitSlice and ObsSlice are immediately available.
func PredictNN(fileRoot string, bSize int, p Pipeline, opts ...NNOpts) (nn *NNModel, err error) {

	nn, err = LoadNN(fileRoot, bSize)
	for _, o := range opts {
		o(nn)
	}
	if err != nil {
		return
	}
	for !p.Batch(nn.Inputs()) {
	}
	vms := G.NewTapeMachine(nn.G())
	defer func() { err = vms.Close() }()
	if err = vms.RunAll(); err != nil {
		return
	}
	return
}

// ByFormula returns model features/targets as FTypes.  The first entry is the target.
// The model is specified as "target~feature1+feature2+...+featurek).  Embeddings are
// specified as "E(<feature name>,<# embedding columns>).
func ByFormula(model string, p Pipeline) (modSpec FTypes, err error) {
	modSpec = make([]*FType, 0)
	err = nil

	var feat *FType
	model = strings.ReplaceAll(model, " ", "")
	lr := strings.Split(model, "~")
	if len(lr) != 2 {
		err = fmt.Errorf("bad formula ~")
		return
	}
	// target
	feat = p.GetFType(lr[0])
	if feat == nil {
		return nil, fmt.Errorf("feature %s not found", lr[0])
	}
	modSpec = append(modSpec, feat)

	fs := strings.Split(lr[1], "+")
	for _, f := range fs {
		ft := f
		embCols := 0
		if strings.Contains(f, "E(") || strings.Contains(f, "e(") {
			l := strings.Split(ft, ",")
			if len(l) != 2 {
				return nil, fmt.Errorf("parse error")
			}
			ft = l[0][2:]
			var em int64
			em, err = strconv.ParseInt(l[1][0:len(l[1])-1], 10, 32)
			if err != nil {
				return nil, err
			}
			if em <= 1 {
				return nil, fmt.Errorf("embedding columns must be at least 2")
			}
			embCols = int(em)
		}
		feat = p.GetFType(ft)
		if feat == nil {
			return nil, fmt.Errorf("feature %s not found", f)
		}
		if feat.Role == FRCat {
			return nil, fmt.Errorf("feature %s is categorical--must convert to one-hot", feat.Name)
		}
		feat.EmbCols = embCols
		if embCols > 0 {
			if feat.Role != FROneHot {
				return nil, fmt.Errorf("feature %s must be one-hot", ft)
			}
			feat.Role = FREmbed
		}
		modSpec = append(modSpec, feat)
	}
	return
}
