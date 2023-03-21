package seafan

// diags.go implements model diagnostics

import (
	"fmt"
	"math"
	"sort"

	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"gonum.org/v1/gonum/stat"
)

const thresh = 0.5 // threshold for declaring y[i] to be a 1

// UnNormalize un-normalizes a slice, if need be
func UnNormalize(vals []float64, ft *FType) (unNorm []float64) {
	outVal := vals
	if ft != nil && ft.Normalized {
		for ind, v := range vals {
			outVal[ind] = v*ft.FP.Scale + ft.FP.Location
		}
	}

	return outVal
}

// Coalesce combines columns of either a one-hot feature or a softmax output.  In the case of a feature,
// it returns 1 if any of the target columns is 1.  In the case of a softmax output, it sums the entries.
func Coalesce(vals []float64, nCat int, trg []int, binary, logodds bool, sl Slicer) ([]float64, error) {

	if nCat < 1 {
		return nil, Wrapper(ErrDiags, "Coalesce: nCat must be at least 1")
	}

	if trg == nil {
		return nil, Wrapper(ErrDiags, "Coalesce: trg cannot be nil")
	}

	if len(vals)%nCat != 0 {
		return nil, Wrapper(ErrDiags, "Coalesce: len y not multiple of nCat")
	}

	for _, t := range trg {
		if t > nCat-1 {
			return nil, Wrapper(ErrDiags, "Coalesce: trg index out of range")
		}
	}
	if binary && logodds {
		return nil, Wrapper(ErrDiags, "coalesce cannot have both binary and logodds")
	}

	n := len(vals) / nCat // # of observations
	coalesced := make([]float64, 0)

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
				den += math.Exp(vals[ind+col])
			}
		}

		outVal := 0.0
		// We may be aggregating over categories of softmax
		for _, col := range trg {
			switch binary {
			case true:
				if vals[ind+col] > thresh {
					outVal = 1.0
				}
			case false:
				switch logodds {
				case true:
					outVal += math.Exp(vals[ind+col]) / den
				case false:
					outVal += vals[ind+col]
				}
			}
		}

		coalesced = append(coalesced, outVal)
	}

	return coalesced, nil
}

// KS finds the KS of a softmax model that is reduced to a binary outcome.
//
//	xy        XY struct where x is fitted value and y is the binary observed value
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

// SegPlot generates a decile plot of the fields y and fit in pipe.  The segments are based on the values of the field seg.
// If seg is continuous, the segments are based on quantiles: 0-.1, .1-.25, .25-.5, .5-.75, .9-1
//
//		obs       observed field (y-axis) name
//		fit       fitted field (x-axis) name
//	    seg       segmenting field name
//		plt       PlotDef plot options.  If plt is nil an error is generated.
func SegPlot(pipe Pipeline, obs, fit, seg string, plt *PlotDef, minVal, maxVal *float64) error {
	const minCnt = 1 // min # of obs for each point

	if plt == nil {
		return Wrapper(ErrDiags, "Decile: plt cannot be nil")
	}

	fitFtype := pipe.GetFType(fit)
	if fitFtype == nil {
		return Wrapper(ErrDiags, fmt.Sprintf("no such field: %s", fit))
	}

	obsFit := pipe.GetFType(obs)
	if obsFit == nil {
		return Wrapper(ErrDiags, fmt.Sprintf("no such field: %s", obs))
	}

	if fitFtype.Role != FRCts || obsFit.Role != FRCts {
		return Wrapper(ErrDiags, "decile Inputs must be type FRCts")
	}

	sliceGrp, e := NewSlice(seg, minCnt, pipe, nil)
	if e != nil {
		return e
	}

	fig := &grob.Fig{}
	minV, maxV := math.MaxFloat64, -math.MaxFloat64
	ind, mad, rowTot := 0, float64(0), float64(0)
	bias := pipe.Get(fit).Summary.DistrC.Mean - pipe.Get(obs).Summary.DistrC.Mean

	for sliceGrp.Iter() {
		slicer := sliceGrp.MakeSlicer()
		pipeSlice, e := pipe.Slice(slicer)
		if e != nil {
			continue
		}
		nSqrt := math.Sqrt(float64(pipeSlice.Rows()))

		distr := pipeSlice.Get(obs).Summary.DistrC
		obsMean, obsStd := distr.Mean, distr.Std/nSqrt
		fitMean := pipeSlice.Get(fit).Summary.DistrC.Mean - bias

		mad += math.Abs(fitMean - obsMean)
		rowTot++
		ci := []float64{obsMean - 2.0*obsStd, obsMean + 2.0*obsStd}
		maxV = math.Max(maxV, ci[1])
		minV = math.Min(minV, ci[0])
		ind++

		trCI := &grob.Scatter{
			Type:       grob.TraceTypeScatter,
			X:          []float64{fitMean, fitMean},
			Y:          ci,
			Name:       fmt.Sprintf("%d: %v", pipeSlice.Rows(), sliceGrp.Value()),
			Hoverlabel: &grob.ScatterHoverlabel{Namelength: -1},
			Mode:       grob.ScatterModeLines,
			Line:       &grob.ScatterLine{Color: "black"},
		}
		fig.AddTraces(trCI)

		tr := &grob.Scatter{
			Type:       grob.TraceTypeScatter,
			X:          []float64{fitMean},
			Y:          []float64{obsMean},
			Name:       fmt.Sprintf("%v", sliceGrp.Value()),
			Hoverlabel: &grob.ScatterHoverlabel{Namelength: -1},
			Mode:       grob.ScatterModeMarkers,
			Line:       &grob.ScatterLine{Color: "green"},
		}
		fig.AddTraces(tr)
	}

	// if user has supplied graph limits, use them
	if minVal != nil {
		minV = *minVal
	}
	if maxVal != nil {
		maxV = *maxVal
	}

	tr := &grob.Scatter{
		Type: grob.TraceTypeScatter,
		X:    []float64{minV, maxV},
		Y:    []float64{minV, maxV},
		Name: "ref",
		Mode: grob.ScatterModeLines,
		Line: &grob.ScatterLine{Color: "red"},
	}
	fig.AddTraces(tr)

	mad /= rowTot
	plt.STitle = fmt.Sprintf("MAD (unbiased fit): %0.4f Bias: %0.4f", mad, bias)

	if plt.XTitle == "" {
		plt.XTitle = fit
	}

	if plt.YTitle == "" {
		plt.YTitle = obs
	}

	if plt.Title == "" {
		plt.Title = "Decile Plot"
	}
	plt.Title = fmt.Sprintf("%s<br>%s", plt.Title, "Bias Corrected")

	err := Plotter(fig, &grob.Layout{}, plt)

	return err
}

// Decile generates a decile plot based on xy
//
//	XY        values to base the plot on.
//	plt       PlotDef plot options.  If plt is nil an error is generated.
//
// The deciles are created based on the values of xy.X
func Decile(xyIn *XY, plt *PlotDef) error {
	if plt == nil {
		return Wrapper(ErrDiags, "Decile: plt cannot be nil")
	}

	// preserve input data by making a copy
	xCopy := make([]float64, xyIn.Len())
	yCopy := make([]float64, xyIn.Len())
	copy(xCopy, xyIn.X)
	copy(yCopy, xyIn.Y)
	xy, e := NewXY(xCopy, yCopy)
	if e != nil {
		return e
	}

	if e = xy.Sort(); e != nil {
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
	minVal, maxVal := math.MaxFloat64, -math.MaxFloat64

	for g := 0; g < ng; g++ {
		nFloat := float64(nDec[g])
		w := math.Sqrt(yDec[g] * (1.0 - yDec[g]) / nFloat)
		lower[g] = yDec[g] - 2.0*w
		upper[g] = yDec[g] + 2.0*w

		minVal = math.Min(math.Min(minVal, lower[g]), fDec[g])
		maxVal = math.Max(math.Max(maxVal, upper[g]), fDec[g])

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
		X:    []float64{minVal, maxVal},
		Y:    []float64{minVal, maxVal},
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

// AddFitted addes fitted values to a Pipeline. The features can be re-normalized/re-mapped to align pipeIn with
// the model build
// pipeIn -- input Pipeline to run the model on
// nnFile -- root directory of NNModel
// target -- target columns of the model output to coalesce
// name -- name of fitted value in Pipeline
// fts -- options FTypes to use for normalizing pipeIn
func AddFitted(pipeIn Pipeline, nnFile string, target []int, name string, fts FTypes, logodds bool, obsFit *FType) error {
	// operate on all data
	bSize := pipeIn.BatchSize()
	WithBatchSize(0)(pipeIn) // all rows
	nn1, e := PredictNNwFts(nnFile, pipeIn, false, fts)
	if e != nil {
		return e
	}

	// Coalesce the output
	bigFit := nn1.FitSlice()

	fit := make([]float64, pipeIn.Rows())
	outCols := nn1.outCols // nn1.Cols()
	for row := 0; row < len(fit); row++ {
		for _, col := range target {
			fit[row] += bigFit[row*outCols+col]
		}

		if logodds {
			switch {
			case fit[row] < 0.0:
				return Wrapper(ErrDiags, "attempt to take log odds of value <0")
			case fit[row] == 0.0:
				fit[row] = -10.0
			case fit[row] > 0.0 && fit[row] < 1.0:
				fit[row] = math.Log(fit[row] / (1.0 - fit[row]))
			case fit[row] == 1.0:
				fit[row] = 10.0
			case fit[row] > 1.0:
				return Wrapper(ErrDiags, "attempt to take log odds of value >1")
			}
		}
	}

	gData := pipeIn.GData()
	fitRaw := NewRawCast(UnNormalize(fit, obsFit), nil)

	if e := gData.AppendField(fitRaw, name, FRCts, pipeIn.GetKeepRaw()); e != nil {
		return e
	}

	WithBatchSize(bSize)(pipeIn)

	return nil
}

// Marginal produces a set of plots to aid in understanding the effect of a feature.
// The plot takes the model output and creates six segments based on the quantiles of the model output:
// (<.1, .1-.25, .25-.5, .5-.75, .75-.9, .9-1).
//
// For each segment, the feature being analyzed various across its range within the quartile (continuous)
// its values (discrete).
// The bottom row shows the distribution of the feature within the quartile range.
func Marginal(nnFile string, feat string, target []int, pipe Pipeline, pd *PlotDef, obsFtype *FType) error {
	const (
		take    = 1000 // # of obs to use for graph
		maxCats = 10   // max # of levels of a categorical field to show in plot
	)
	var e error

	name := feat
	lay := &grob.Layout{}
	lay.Grid = &grob.LayoutGrid{Rows: 2, Columns: 6, Pattern: grob.LayoutGridPatternIndependent, Roworder: grob.LayoutGridRoworderTopToBottom}
	fig := &grob.Fig{}

	bSize := pipe.BatchSize()
	defer WithBatchSize(bSize)(pipe)

	WithBatchSize(pipe.Rows())(pipe)

	if e = AddFitted(pipe, nnFile, target, "fitted", nil, false, obsFtype); e != nil {
		return Wrapper(e, "Marginal")
	}

	targFt := pipe.Get(feat) // feature we're working on
	if targFt == nil {
		return Wrapper(ErrDiags, fmt.Sprintf("Marginal: feature %s not in model", feat))
	}

	slice, e := NewSlice("fitted", 0, pipe, nil)
	if e != nil {
		return Wrapper(e, "Marginal")
	}

	plotNo := 12 // used as a basis to know which plot we're working on

	for slice.Iter() {
		sl := slice.MakeSlicer()
		newPipe, e := pipe.Slice(sl)
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
			keys, vals := gdFrom.Summary.DistrD.Sort(false, false)

			// convert counts to rates
			rate := make([]float64, len(vals))
			for ind := 0; ind < len(vals); ind++ {
				rate[ind] = float64(vals[ind]) / float64(gd.Summary.NRows)
			}

			cats := Min(len(keys), maxCats)
			tr := &grob.Bar{Xaxis: xAxis, Yaxis: yAxis, X: keys[0:cats], Y: rate[0:cats], Type: grob.TraceTypeBar}

			fig.AddTraces(tr)

			nper := n / cats
			data := gd.Data

			for row := 0; row < n; row++ {
				grp := Min(row/nper, cats-1)
				grpKey := keys[grp]
				grpVal := int(gdFrom.FT.FP.Lvl[grpKey])

				for c := 0; c < cats; c++ {
					data.([]float64)[row*cats+c] = 0.0
				}

				data.([]float64)[row*cats+grpVal] = 1.0
				xs1[row] = fmt.Sprintf("%v", grpKey)
			}
		default:
			return Wrapper(ErrDiags, fmt.Sprintf("Marginal: feature %s is discrete -- need OneHot", feat))
		}

		// predict on data we just created
		nn1, e := PredictNN(nnFile, newPipe, false)
		if e != nil {
			return Wrapper(e, "Marginal")
		}

		nCat := nn1.OutputCols() // nn1.Cols()

		fit, e := Coalesce(UnNormalize(nn1.FitSlice(), obsFtype), nCat, target, false, false, nil)
		if e != nil {
			return Wrapper(e, "Marginal")
		}

		xAxis, yAxis = fmt.Sprintf("x%d", plotNo-6), fmt.Sprintf("y%d", plotNo-6)
		plotNo--
		tr := &grob.Box{X: xs1, Y: fit, Type: grob.TraceTypeBox, Xaxis: xAxis, Yaxis: yAxis}

		fig.AddTraces(tr)
		if plotNo == 6 {
			break
		}
	}

	pd.Title = fmt.Sprintf("Marginal Effect of %s by Quartile of Fitted Value (High to Low)<br>%s", name, pd.Title)
	if e := Plotter(fig, lay, pd); e != nil {
		return Wrapper(e, "Marginal")
	}

	return nil
}

// R2 returns the model r-square.  Returns -1 if an error.
func R2(y, yhat []float64) float64 {
	if len(y) != len(yhat) {
		return -1
	}

	my := stat.Mean(y, nil)
	tss := 0.0
	sse := 0.0
	for ind := 0; ind < len(y); ind++ {
		res := y[ind] - my
		tss += res * res
		res = y[ind] - yhat[ind]
		sse += res * res
	}

	if tss == 0.0 {
		return -1
	}

	return 100.0 * (1.0 - sse/tss)
}
