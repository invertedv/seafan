package seafan

// nn.go implements NN functionality

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gorgonia.org/golgi"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// CostFunc function prototype for cost functions
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
	construct ModSpec      // model spec
	costFn    CostFunc     // costFn corresponding to cost *G.Node
	build     bool         // build mode includes drop out layers
	inputFT   FTypes       // FTypes of input features
	targetFT  *FType       // FType of output (target)
	outCols   int          // columns in output
}

// Name returns model name
func (m *NNModel) Name() string {
	return m.name
}

// Cols returns # of columns in NNModel output
func (m *NNModel) Cols() int {
	return m.outCols
}

func (m *NNModel) InputFT() FTypes {
	return m.inputFT
}

func (m *NNModel) String() string {
	if m.construct == nil {
		return "No model"
	}

	str := fmt.Sprintf("%s\nInputs\n", m.Name())

	for _, ft := range m.inputFT {
		str = fmt.Sprintf("%s%v\n", str, ft)
	}

	str = fmt.Sprintf("%sTarget\n", str)

	switch m.targetFT == nil {
	case true:
		str = fmt.Sprintf("%sNone\n", str)
	case false:
		str = fmt.Sprintf("%s%v", str, m.targetFT)
	}

	str = fmt.Sprintf("%s\nModel Structure\n", str)

	for ind := 0; ind < len(m.construct); ind++ {
		str = fmt.Sprintf("%s%s\n", str, m.construct[ind])
	}

	str = fmt.Sprintf("%s\n", str)

	if m.cost != nil {
		str = fmt.Sprintf("%sCost function: %s\n\n", str, m.cost.Name())
	}

	bSize := m.inputsC[0].Shape()[0]
	str = fmt.Sprintf("%sBatch size: %d\n", str, bSize)

	nPar := 0
	for _, n := range m.paramsW {
		nPar += n.Shape()[0] * n.Shape()[1]
	}

	for _, n := range m.paramsB {
		nPar += n.Shape()[0] * n.Shape()[1]
	}

	str = fmt.Sprintf("%s%d FC parameters\n", str, nPar)
	nEmb := 0

	for _, n := range m.paremsEmb {
		nEmb += n.Shape()[0] * n.Shape()[1]
	}

	str = fmt.Sprintf("%s%d Embedding parameters\n", str, nEmb)

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
	if m.obs == nil {
		return nil
	}
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

// OutputCols returns the number of columns in the output
func (m *NNModel) OutputCols() int {
	return m.output.Nodes()[0].Shape()[1]
}

// Inputs returns input (continuous+embedded+observed) inputs
func (m *NNModel) Inputs() G.Nodes {
	n := append(m.inputsC, m.inputsE...)

	if m.obs == nil {
		return n
	}

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

// NNOpts -- NNModel options
type NNOpts func(model1 *NNModel)

// WithCostFn adds a cost function
func WithCostFn(cf CostFunc) NNOpts {
	f := func(m *NNModel) {
		m.costFn = cf
		m.cost = cf(m)
	}

	return f
}

// WithName adds a name to the NNModel
func WithName(name string) NNOpts {
	f := func(m *NNModel) {
		m.name = name
	}

	return f
}

// NewNNModel creates a new NN model.
// Specs for fields in modSpec are pulled from pipe.
// if build is true, DropOut layers are included.
func NewNNModel(modSpec ModSpec, pipe Pipeline, build bool, no ...NNOpts) (*NNModel, error) {
	bSize := pipe.BatchSize()
	g := G.NewGraph()
	xs := make(G.Nodes, 0)
	embParm := make(G.Nodes, 0) // embedding parameters
	xEmInp := make(G.Nodes, 0)  // one-hot input
	xEmProd := make(G.Nodes, 0) // product of one-hot input and embedding parameters
	// work through the features
	inps, e := modSpec.Inputs(pipe)
	if e != nil {
		return nil, e
	}

	for ind := 0; ind < len(inps); ind++ {
		f := inps[ind]
		// first element is the target--skip
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
		zemb := G.Must(G.Concat(1, xEmProd...)) // embeddings for input to FCLayer layer
		xall = G.Must(G.Concat(1, xall, zemb))
	}

	// target.  There may not be a target if the model has been built and is now in prediction mode.
	obsF, e := modSpec.Target(pipe)
	if e != nil {
		return nil, e
	}

	var yoh *G.Node
	yoh = nil

	if obsF != nil {
		switch obsF.Role {
		case FRCts:
			yoh = G.NewTensor(g, tensor.Float64, 2, G.WithName(obsF.Name), G.WithShape(bSize, 1))
		case FROneHot:
			yoh = G.NewTensor(g, tensor.Float64, 2, G.WithName(obsF.Name), G.WithShape(bSize, obsF.Cats))
		default:
			return nil, Wrapper(ErrNNModel, "NewNNModel: output must be either FRCts or FROneHot")
		}
	}

	lastCols := xall.Shape()[1] // layer output dim
	parW := make(G.Nodes, 0)
	parB := make(G.Nodes, 0)

	adder := 0 // add 1 if the output is softmax
	for ind := 0; ind < len(modSpec); ind++ {
		ly, e := modSpec.LType(ind)
		if e != nil {
			return nil, e
		}

		if *ly != FC {
			continue
		}

		fc := modSpec.FC(ind)

		if fc == nil {
			return nil, Wrapper(ErrNNModel, fmt.Sprintf("NewNNModel: error parsing layer %d", ind))
		}

		cols := fc.Size

		if fc.Act == SoftMax {
			if obsF != nil && obsF.Role != FROneHot {
				return nil, Wrapper(ErrNNModel, "NewNNModel: obs not one-hot but softmax activation")
			}
			cols--
			adder = 1
		} else {
			adder = 0
		}

		nmw := "lWeights" + strconv.Itoa(ind)
		w := G.NewTensor(g, tensor.Float64, 2, G.WithName(nmw), G.WithShape(lastCols, cols), G.WithInit(G.GlorotN(1.0)))

		if fc.Bias {
			nmb := "lBias" + strconv.Itoa(ind)
			b := G.NewTensor(g, tensor.Float64, 2, G.WithName(nmb), G.WithShape(1, cols), G.WithInit(G.GlorotN(1.0)))
			parB = append(parB, b)
		}

		lastCols = cols

		parW = append(parW, w)
	}

	outputCols := lastCols + adder

	if yoh != nil {
		if yoh.Shape()[1] != outputCols {
			return nil, Wrapper(ErrNNModel, "NewNNModel: output node and obs node have differing columns")
		}
	}

	nn := &NNModel{
		g:         g,
		paramsW:   parW,
		paramsB:   parB,
		paremsEmb: embParm,
		inputsC:   xs,
		inputsE:   xEmInp,
		obs:       yoh,
		construct: modSpec,
		build:     build,
		inputFT:   inps,
		targetFT:  obsF,
		outCols:   outputCols,
	}

	nn.Fwd() // init forward pass
	// add user opts
	for _, o := range no {
		o(nn)
	}

	return nn, nil
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

	out := xall

	// work through layers
	for ind := 1; ind < len(m.construct); ind++ {
		ltype, e := m.construct.LType(ind)
		if e != nil {
			panic(e)
		}

		switch *ltype {
		case FC:
			fc := m.construct.FC(ind)
			nmw := "lWeights" + strconv.Itoa(ind)

			px := GetNode(m.paramsW, nmw)
			out = G.Must(G.Mul(out, px))
			nmb := "lBias" + strconv.Itoa(ind)
			bias := GetNode(m.paramsB, nmb)

			if bias != nil {
				out = G.Must(G.BroadcastAdd(out, bias, nil, []byte{0}))
			}

			switch fc.Act {
			case Relu:
				out = ReluAct(out)
			case LeakyRelu:
				out = LeakyReluAct(out, fc.ActParm)
			case Sigmoid:
				out = SigmoidAct(out)
			case SoftMax:
				out = SoftMaxAct(out)
			}
		case DropOut:
			if m.build {
				if d := m.construct.DropOut(ind); d != nil {
					out = G.Must(G.Dropout(out, d.DropProb))
				}
			}
		}
	}

	m.output = out
}

// struct to save nodes to json file
type saveNode struct {
	Name  string    `json:"name"`
	Dims  []int     `json:"dims"`
	Parms []float64 `json:"parms"`
}

// Save saves a model to disk.  Two files are created: <fileRoot>S.nn for the ModSpec and
// <fileRoot>P.nn form the parameters.
func (m *NNModel) Save(fileRoot string) (err error) {
	err = nil
	fileP := fileRoot + "P.nn"
	f, err := os.Create(fileP)

	if err != nil {
		return
	}

	defer func() { _ = f.Close() }()

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

// LoadNN restores a previously saved NNModel.
// fileRoot is the root name of the save file.
// p is the Pipeline with the field specs.
// if build is true, DropOut layers are included.
func LoadNN(fileRoot string, p Pipeline, build bool) (nn *NNModel, err error) {
	err = nil
	nn = nil
	fileS := fileRoot + "S.nn"
	modSpec, err := LoadModSpec(fileS)

	if err != nil {
		return
	}

	fileP := fileRoot + "P.nn"
	f, err := os.Open(fileP)

	if err != nil {
		return
	}

	defer func() { _ = f.Close() }()

	js, err := io.ReadAll(f)

	if err != nil {
		return
	}

	data := make([]saveNode, 0)
	if e := json.Unmarshal(js, &data); e != nil {
		return nil, e
	}

	nn, err = NewNNModel(modSpec, p, build)
	if err != nil {
		return nil, err
	}

	if len(data) != len(nn.Params()) {
		return nil, Wrapper(ErrNNModel, "LoadNN: node count differs")
	}

	for _, d := range data {
		nd := nn.g.ByName(d.Name)[0]
		if nd == nil {
			return nil, Wrapper(ErrNNModel, fmt.Sprintf("LoadNN: node %s not found", d.Name))
		}

		shp := nd.Shape()

		for ind, dim := range shp {
			if dim != d.Dims[ind] {
				return nil, Wrapper(ErrNNModel, "LoadNN: dimensions do not match")
			}
		}

		t := tensor.New(tensor.WithBacking(d.Parms), tensor.WithShape(shp...))

		if err := G.Let(nd, t); err != nil {
			return nil, err
		}
	}
	inps, e := modSpec.Inputs(p)
	if e != nil {
		return nil, Wrapper(e, "LoadNN")
	}
	nn.inputFT = inps

	return nn, nil
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
	shuffle   int
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
		shuffle: 0,
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

// WithShuffle shuffles after interval epochs
// Default is 0 (don't shuffle ever)
func WithShuffle(interval int) FitOpts {
	f := func(ft *Fit) {
		ft.shuffle = interval
	}

	return f
}

// WithLearnRate sets a learning rate function that declines linearly across the epochs.
func WithLearnRate(lrStart, lrEnd float64) FitOpts {
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

// Do is the fitting loop.
func (ft *Fit) Do() (err error) {
	best := math.MaxFloat64
	ft.bestEpoch = 0

	if _, e := G.Grad(ft.nn.Cost(), ft.nn.Params()...); e != nil {
		panic(e)
	}

	vm := G.NewTapeMachine(ft.nn.G(), G.BindDualValues(ft.nn.Params()...))

	defer func() { _ = vm.Close() }()

	t := time.Now()
	itv := make([]float64, 0)
	solv := G.NewAdamSolver()

	if ft.l2Penalty > 0.0 {
		G.WithL2Reg(ft.l2Penalty)(solv)
	}

	cv := make([]float64, 0)
	cVal := make([]float64, 0)
	cte := true
	for ep := 1; ep <= ft.epochs && cte; ep++ {
		if ft.shuffle > 0 && ep%ft.shuffle == 0 {
			ft.p.Shuffle()
		}
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
			// find validation cost, save model and load to new graph
			if e := ft.nn.Save(ft.tmpFile); err != nil {
				return e
			}

			var valMod *NNModel
			// with a validation set, don't use dropouts
			valMod, err = PredictNN(ft.tmpFile, ft.pVal, false, WithCostFn(ft.nn.CostFn()))
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
			if ft.wait > 0 && ep-ft.bestEpoch > ft.wait {
				cte = false
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
	return nil
}

// PredictNN reads in a NNModel from a file and populates it with a batch from p.
// Methods such as FitSlice and ObsSlice are immediately available.
func PredictNN(fileRoot string, pipe Pipeline, build bool, opts ...NNOpts) (nn *NNModel, err error) {
	nn, err = LoadNN(fileRoot, pipe, build)
	for _, o := range opts {
		o(nn)
	}

	if err != nil {
		return
	}

	for !pipe.Batch(nn.Inputs()) {
	}

	vms := G.NewTapeMachine(nn.G())

	defer func() { _ = vms.Close() }()

	if err = vms.RunAll(); err != nil {
		return
	}

	return
}

// PredictNNwFts updates the input pipe to have the FTypes specified by fts. For instance, if one has normalized a
// continuous input, the normalization factor used in the NN must be the same as its build values.
func PredictNNwFts(fileRoot string, pipe Pipeline, build bool, fts FTypes, opts ...NNOpts) (nn *NNModel, err error) {
	if fts == nil {
		return PredictNN(fileRoot, pipe, build, opts...)
	}

	gd := pipe.GData()
	newGd, e := gd.UpdateFts(fts)
	if e != nil {
		return nil, e
	}

	// if something is in here as a FRCat, FREmbed then we need to add a one-hot field
	for _, fld := range newGd.FieldList() {
		ft := newGd.Get(fld).FT
		if ft.Role == FRCat || ft.Role == FREmbed {
			if e = newGd.MakeOneHot(ft.Name, ft.Name+"Oh"); e != nil {
				return nil, e
			}
		}
	}

	vecPipe := NewVecData("predict with FTypes", newGd, WithBatchSize(pipe.BatchSize()))

	return PredictNN(fileRoot, vecPipe, build, opts...)
}

// SoftMaxAct implements softmax activation functin
func SoftMaxAct(n *G.Node) *G.Node {
	exp := G.Must(G.Exp(n))
	sexp := G.Must(G.Sum(exp, 1))
	sexp = G.Must(G.Add(sexp, G.NewConstant(1.0)))
	phat := G.Must(G.BroadcastHadamardDiv(exp, sexp, nil, []byte{1}))
	phats := G.Must(G.Sum(phat, 1))
	phat1 := G.Must(G.Sub(G.NewConstant(1.0), phats))
	r := phat1.Shape()[0]
	phat1a := G.Must(G.Reshape(phat1, tensor.Shape{r, 1}))
	phat2 := G.Must(G.Concat(1, phat, phat1a))

	return phat2
}

// LinearAct is a no-op.  It is the default ModSpec default activation.
func LinearAct(n *G.Node) *G.Node {
	return n
}

// ReluAct is relu activation
func ReluAct(n *G.Node) *G.Node {
	return G.Must(G.LeakyRelu(n, 0.0))
}

// LeakyReluAct is leaky relu activation
func LeakyReluAct(n *G.Node, alpha float64) *G.Node {
	return G.Must(G.LeakyRelu(n, alpha))
}

// SigmoidAct is sigmoid activation
func SigmoidAct(n *G.Node) *G.Node {
	return G.Must(G.Sigmoid(n))
}

// GetNode returns a node by name from a G.Nodes
func GetNode(ns G.Nodes, name string) *G.Node {
	for _, n := range ns {
		if n.Name() == name {
			return n
		}
	}

	return nil
}
