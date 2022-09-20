package seafan

// diags.go implements model diagnostics

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"gonum.org/v1/gonum/stat"
	"math"
	"sort"
)

const thresh = 0.5 // threshold for declaring y[i] to be a 1

// Coalesce reduces a softmax output to two categories
//
//	y         observed multinomial values
//	fit       softmax fit to y
//	nCat      # of categories
//	trg       columns of y to be grouped into a single outcome. The complement is reduced to the alternate outcome.
//	logodds   if true, fit is in log odds space
//
// An XY struct of the coalesced outcome (Y) & fitted values (X) is returned
func Coalesce(y, fit []float64, nCat int, trg []int, logodds bool, sl Slicer) (*XY, error) {

	if len(y) != len(fit) {
		return nil, Wrapper(ErrDiags, "Coalesce: y and fit must have same len")
	}

	if nCat < 1 {
		return nil, Wrapper(ErrDiags, "Coalesce: nCat must be at least 1")
	}

	if trg == nil {
		return nil, Wrapper(ErrDiags, "Coalesce: trg cannot be nil")
	}

	if len(y)%nCat != 0 {
		return nil, Wrapper(ErrDiags, "Coalesce: len y not multiple of nCat")
	}

	for _, t := range trg {
		if t > nCat-1 {
			return nil, Wrapper(ErrDiags, "Coalesce: trg index out of range")
		}
	}

	n := len(y) / nCat // # of observations
	xOut := make([]float64, 0)
	yOut := make([]float64, 0)

	for row := 0; row < n; row++ {
		if sl != nil && !sl(row) {
			continue
		}
		// index into y/pred which is stored by row
		ind := row * nCat
		den := 1.0
		// if the input is log odds, we need to reconstruct probabilities
		if logodds {
			den = 0.0
			for col := 0; col < nCat; col++ {
				den += math.Exp(fit[ind+col])
			}
		}

		pred, obs := 0.0, 0.0 // predicted & observed
		// We may be aggregating over categories of softmax
		for _, col := range trg {
			// y is one if any of the trg levels is 1
			if y[ind+col] > thresh {
				obs = 1.0
			}

			switch logodds {
			case true:
				pred += math.Exp(fit[ind+col]) / den
			case false:
				pred += fit[ind+col]
			}
		}

		xOut = append(xOut, pred)
		yOut = append(yOut, obs)
	}

	return &XY{X: xOut, Y: yOut}, nil
}

// KS finds the KS of a softmax model that is reduced to a binary outcome.
//
//	y         observed multinomial values
//	fit       fitted softmax probabilities
//	trg       columns of y to be grouped into a single outcome. The complement is reduced to the alternate outcome.
//	logodds   if true, fit is in log odds space
//	plt       PlotDef plot options.  If plt is nil, no plot is produced.
//
// The ks statistic is returned as are Desc descriptions of the model for the two groups.
// Returns
//
//	ks          KS statistic
//	notTarget  Desc struct of fitted values of the non-target outcomes
//	target     Desc struct of fitted values of target outcomes
//
// Target: html plot file and/or plot in browser.
func KS(xy *XY, plt *PlotDef) (ks float64, notTarget *Desc, target *Desc, err error) {
	const nPoints = 101 // # of points for ks plot
	const divisor = float64(nPoints - 1)

	n := len(xy.X)
	// arrays to hold probabilities with observed 0's and 1's separately
	probNotTarget, probTarget := make([]float64, 0), make([]float64, 0)

	for row := 0; row < n; row++ {
		// append to appropriate slice
		switch {
		case xy.Y[row] > thresh:
			probTarget = append(probTarget, xy.X[row])
		default:
			probNotTarget = append(probNotTarget, xy.X[row])
		}
	}

	notTarget, _ = NewDesc(nil, "not target") // fmt.Sprintf("Value not in %v", trg))
	target, _ = NewDesc(nil, "target")        // fmt.Sprintf("Value in %v", trg))

	notTarget.Populate(probNotTarget, false, nil) // side effect is probNotTarget is sorted
	target.Populate(probTarget, false, nil)

	// Min & max of probabilities
	lower := math.Min(notTarget.Q[0], target.Q[0])
	upper := math.Max(notTarget.Q[len(notTarget.Q)-1], target.Q[len(target.Q)-1])

	// p is an array that is equally spaced between lower and upper
	p := make([]float64, nPoints)
	// these are cumulative distributions
	for k := 0; k < nPoints; k++ {
		p[k] = (float64(k)/divisor)*(upper-lower) + lower
	}
	sort.Float64s(probNotTarget)
	sort.Float64s(probTarget)

	upnt := make([]float64, len(probNotTarget))

	for k := 0; k < len(upnt); k++ {
		upnt[k] = float64(k+1) / float64(len(upnt))
	}

	upt := make([]float64, len(probTarget))

	for k := 0; k < len(upt); k++ {
		upt[k] = float64(k+1) / float64(len(upt))
	}

	xypt, _ := NewXY(probTarget, upt)
	xypt.X = append(xypt.X, 0, 1)
	xypt.Y = append(xypt.Y, 0, 1)
	xypnt, _ := NewXY(probNotTarget, upnt)
	xypnt.X = append(xypnt.X, 0, 1)
	xypnt.Y = append(xypnt.Y, 0, 1)
	cpt, _ := xypt.Interp(p)
	cpnt, _ := xypnt.Interp(p)

	ks, at := 0.0, 0.0
	cumeNotTarget := cpnt.Y
	cumeTarget := cpt.Y

	for k := 0; k < nPoints; k++ {
		if d := 100.0 * math.Abs(cumeTarget[k]-cumeNotTarget[k]); d > ks {
			ks = d
			at = p[k]
		}
	}

	// plot, if requested
	if plt != nil {
		t0 := &grob.Scatter{
			Type: grob.TraceTypeScatter,
			X:    p,
			Y:    cumeNotTarget,
			Name: notTarget.Name,
			Mode: grob.ScatterModeLines,
			Line: &grob.ScatterLine{Color: "black"},
		}
		t1 := &grob.Scatter{
			Type: grob.TraceTypeScatter,
			X:    p,
			Y:    cumeTarget,
			Mode: grob.ScatterModeLines,
			Name: target.Name,
			Line: &grob.ScatterLine{Color: "red"},
		}
		fig := &grob.Fig{Data: grob.Traces{t0, t1}}
		plt.Title = fmt.Sprintf("%s<br>KS %v at %v", plt.Title, math.Round(10.0*ks)/10.0, math.Round(1000*at)/1000)

		if plt.XTitle == "" {
			plt.XTitle = "Fitted Values"
		}

		if plt.YTitle == "" {
			plt.YTitle = "Cumulative Score Distribution"
		}

		if plt.Title == "" {
			plt.Title = "KS Plot"
		}

		lay := &grob.Layout{}
		lay.Legend = &grob.LayoutLegend{X: target.Q[0], Y: 1.0}
		err = Plotter(fig, lay, plt)
	}
	return ks, notTarget, target, err
}

// Decile generates a decile plot of a softmax model that is reduced to a binary outcome.
//
//	y         observed multinomial values
//	fit       fitted softmax probabilities
//	trg       columns of y to be grouped into a single outcome. The complement is reduced to the alternate outcome.
//	logodds   if true, fit is in log odds space
//	plt       PlotDef plot options.  If plt is nil an error is generated.
//
// Target: html plot file and/or plot in browser.
func Decile(xy *XY, plt *PlotDef) error {
	if plt == nil {
		return Wrapper(ErrDiags, "Decile: plt cannot be nil")
	}

	a, b := 0.0, 0.0

	for j := 0; j < xy.Len(); j++ {
		a += xy.X[j]
		b += xy.Y[j]
	}

	if e := xy.Sort(); e != nil {
		return e
	}

	deciles, e := NewDesc([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, "fitted")

	if e != nil {
		return e
	}

	deciles.Populate(xy.X, false, nil)

	ng := len(deciles.U) + 1
	fDec, yDec, nDec := make([]float64, ng), make([]float64, ng), make([]int, ng)

	for row := 0; row < len(xy.X); row++ {
		grp := ng - 1

		for g := 0; g < len(deciles.U); g++ {
			if xy.X[row] < deciles.Q[g] {
				grp = g

				break
			}
		}

		fDec[grp] += xy.X[row]
		yDec[grp] += xy.Y[row]
		nDec[grp]++
	}

	for g := 0; g < ng; g++ {
		if nDec[g] == 0 {
			return Wrapper(ErrDiags, fmt.Sprintf("Decile: decile group %d has no observations", g))
		}

		nFloat := float64(nDec[g])
		fDec[g] /= nFloat
		yDec[g] /= nFloat
	}

	tr := &grob.Scatter{
		Type: grob.TraceTypeScatter,
		X:    fDec,
		Y:    yDec,
		Name: "decile averages",
		Mode: grob.ScatterModeMarkers,
		Line: &grob.ScatterLine{Color: "black"},
	}

	fig := &grob.Fig{Data: grob.Traces{tr}}

	lower, upper := make([]float64, ng), make([]float64, ng)

	for g := 0; g < ng; g++ {
		nFloat := float64(nDec[g])
		w := math.Sqrt(yDec[g] * (1.0 - yDec[g]) / nFloat)
		lower[g] = yDec[g] - 2.0*w
		upper[g] = yDec[g] + 2.0*w

		trCI := &grob.Scatter{
			Type: grob.TraceTypeScatter,
			X:    []float64{fDec[g], fDec[g]},
			Y:    []float64{lower[g], upper[g]},
			Name: fmt.Sprintf("CI%d", g),
			Mode: grob.ScatterModeLines,
			Line: &grob.ScatterLine{Color: "black"},
		}
		fig.AddTraces(trCI)
	}
	tr = &grob.Scatter{
		Type: grob.TraceTypeScatter,
		X:    []float64{0, 1},
		Y:    []float64{0, 1},
		Name: "ref",
		Mode: grob.ScatterModeLines,
		Line: &grob.ScatterLine{Color: "red"},
	}
	fig.AddTraces(tr)

	mFit := stat.Mean(xy.X, nil)
	mObs := stat.Mean(xy.Y, nil)
	n := xy.Len()
	plt.STitle = fmt.Sprintf("95%% CI assuming independence<br># obs: %d means: Fit %0.3f actual %0.3f", n, mFit, mObs)

	if plt.XTitle == "" {
		plt.XTitle = "Fitted Values"
	}

	if plt.YTitle == "" {
		plt.YTitle = "Actual Values"
	}

	if plt.Title == "" {
		plt.Title = "Decile Plot"
	}

	err := Plotter(fig, &grob.Layout{}, plt)

	return err
}

// Assess returns a selection of statistics of the fit
func Assess(xy *XY, cutoff float64) (n int, precision, recall, accuracy float64, obs, fit *Desc, err error) {
	correctYes := 0
	correct := 0
	obsTot := 0
	predTot := 0

	for row := 0; row < len(xy.X); row++ {
		predYes := xy.X[row] > cutoff
		obsYes := xy.Y[row] > 0.999

		if predYes && obsYes {
			correctYes++
			correct++
		}

		if !predYes && !obsYes {
			correct++
		}

		if obsYes {
			obsTot++
		}

		if predYes {
			predTot++
		}
	}

	if obsTot == 0 {
		return 0, 0.0, 0.0, 0.0, nil, nil, Wrapper(ErrDiags, "Decile: there are not positive outcomes")
	}
	if obsTot == xy.Len() {
		return 0, 0.0, 0.0, 0.0, nil, nil, Wrapper(ErrDiags, "Decile: there are not negative outcomes")
	}

	precision = float64(correctYes) / float64(predTot) // fraction of Yes corrects to total predicted Yes
	recall = float64(correctYes) / float64(obsTot)     // fraction of Yes corrects to total actual Yes
	accuracy = float64(correct) / float64(len(xy.X))   // fraction of corrects
	n = len(xy.X)
	fit, err = NewDesc(nil, "fitted values")
	if err != nil {
		return
	}

	fit.Populate(xy.X, true, nil)

	obs, err = NewDesc(nil, "observed values")
	if err != nil {
		return
	}

	obs.Populate(xy.Y, true, nil)
	return n, precision, recall, accuracy, obs, fit, err
}

// AddFitted creates a new Pipeline that adds a NNModel fitted value
func AddFitted(pipeIn Pipeline, nnFile string, target []int) (Pipeline, error) {
	nn1, e := PredictNN(nnFile, pipeIn, false)
	if e != nil {
		panic(e)
	}

	nCat := nn1.Obs().Nodes()[0].Shape()[1]
	xy, e := Coalesce(nn1.ObsSlice(), nn1.FitSlice(), nCat, target, false, nil)
	if e != nil {
		return nil, Wrapper(e, "Marginal")
	}

	gData := pipeIn.GData()
	f120R := NewRawCast(xy.X, nil)
	e = gData.AppendC(f120R, "fitted", false, nil)
	if e != nil {
		return nil, Wrapper(e, "AddFit")
	}

	return NewVecData("with fitted", gData), nil
}

// Marginal produces a set of plots to aid in understanding the effect of a feature.
// The plot takes the model output and creates four segments based on the quartiles of the model output.
// For each segment, the feature being analyzed various across its range within the quartile (continuous)
// its values (discrete).
// The bottom row shows the distribution of the feature within the quartile range.
func Marginal(nnFile string, feat string, target []int, pipe Pipeline, pd *PlotDef) error {
	const take = 1000 // # of obs to use for graph
	var e error

	name := feat
	lay := &grob.Layout{}
	lay.Grid = &grob.LayoutGrid{Rows: 2, Columns: 4, Pattern: grob.LayoutGridPatternIndependent, Roworder: grob.LayoutGridRoworderTopToBottom}
	fig := &grob.Fig{}

	bSize := pipe.BatchSize()
	defer WithBatchSize(bSize)(pipe)

	WithBatchSize(pipe.Rows())(pipe)

	pipeFit, e := AddFitted(pipe, nnFile, target)
	if e != nil {
		return Wrapper(e, "Marginal")
	}

	WithBatchSize(pipeFit.Rows())(pipeFit)

	targFt := pipeFit.Get(feat) // feature we're working on
	if targFt == nil {
		return Wrapper(ErrDiags, fmt.Sprintf("Marginal: feature %s not in model", feat))
	}

	slice, e := NewSlice("fitted", 0, pipeFit, nil)
	if e != nil {
		return Wrapper(e, "Marginal")
	}

	plotNo := 8 // used as a basis to know which plot we're working on

	for slice.Iter() {
		sl := slice.MakeSlicer()
		newPipe, e := pipeFit.Slice(sl)
		if e != nil {
			return Wrapper(e, "Marginal")
		}

		newPipe.Shuffle()

		n := Min(newPipe.Rows(), take)

		WithBatchSize(n)(newPipe)

		xs1 := make([]string, n)
		gd := newPipe.Get(feat)
		// sets the plot to work on:
		xAxis, yAxis := fmt.Sprintf("x%d", plotNo), fmt.Sprintf("y%d", plotNo)

		switch gd.FT.Role {
		case FRCts:
			x := make([]float64, n)

			for ind := 0; ind < n; ind++ {
				x[ind] = gd.Data.([]float64)[ind]*gd.FT.FP.Scale + gd.FT.FP.Location
			}

			tr := &grob.Histogram{Xaxis: xAxis, Yaxis: yAxis, X: x, Type: grob.TraceTypeHistogram}

			fig.AddTraces(tr)

			qs := gd.Summary.DistrC.Q
			dp := (qs[6] - qs[0]) / 5
			nper := n / 4
			data := gd.Data

			for row := 0; row < n; row++ {
				grp := 1 + Min(row/nper, 3)
				xx := qs[0] + dp*float64(grp)
				data.([]float64)[row] = xx
				xs1[row] = fmt.Sprintf("%0.2f", xx*gd.FT.FP.Scale+gd.FT.FP.Location)
			}
		case FROneHot, FREmbed:
			gdFrom := newPipe.Get(gd.FT.From)
			name = gd.FT.From
			key, _ := gdFrom.FT.FP.Lvl.Sort(false, true)
			keyStr := make([]string, len(key))

			for ind := 0; ind < len(key); ind++ {
				keyStr[ind] = fmt.Sprintf("%v", key[ind])
			}

			x := make([]string, n)

			for ind := 0; ind < n; ind++ {
				x[ind] = keyStr[gdFrom.Data.([]int32)[ind]]
			}

			sort.Strings(x)

			tr := &grob.Histogram{Xaxis: xAxis, Yaxis: yAxis, X: x, Type: grob.TraceTypeHistogram}

			fig.AddTraces(tr)

			cats := gd.FT.Cats
			nper := n / cats
			data := gd.Data

			for row := 0; row < n; row++ {
				grp := Min(row/nper, cats-1)

				for c := 0; c < cats; c++ {
					data.([]float64)[row*cats+c] = 0.0
				}

				data.([]float64)[row*cats+grp] = 1.0
				xs1[row] = keyStr[grp]
			}
		default:
			return Wrapper(ErrDiags, fmt.Sprintf("Marginal: feature %s is discrete -- need OneHot", feat))
		}

		// predict on data we just created
		nn1, e := PredictNN(nnFile, newPipe, false)
		if e != nil {
			return Wrapper(e, "Marginal")
		}

		nCat := nn1.Obs().Nodes()[0].Shape()[1]
		xy, e := Coalesce(nn1.ObsSlice(), nn1.FitSlice(), nCat, target, false, nil)
		if e != nil {
			return Wrapper(e, "Marginal")
		}

		xAxis, yAxis = fmt.Sprintf("x%d", plotNo-4), fmt.Sprintf("y%d", plotNo-4)
		plotNo--
		tr := &grob.Box{X: xs1, Y: xy.X, Type: grob.TraceTypeBox, Xaxis: xAxis, Yaxis: yAxis}

		fig.AddTraces(tr)
	}

	pd.Title = fmt.Sprintf("Marginal Effect of %s by Quartile of Fitted Value (Low to High)<br>%s", name, pd.Title)

	if e := Plotter(fig, lay, pd); e != nil {
		return Wrapper(e, "Marginal")
	}

	return nil
}
