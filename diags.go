package seafan

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"math"
)

type Slicer func(row int) bool

// Coalesce reduces a softmax output to two categories
//
//	y         observed multinomial values
//	fit       softmax fit to y
//	nCat      # of categories
//	trg       columns of y to be grouped into a single outcome. The complement is reduced to the alternate outcome.
//	logodds   if true, fit is in log odds space
//
// An XY struct of the coalesced outcome (Y) & fitted values (X) is returned
func Coalesce(y []float64, fit []float64, nCat int, trg []int, logodds bool, sl Slicer) (*XY, error) {
	if len(y) != len(fit) {
		return nil, fmt.Errorf("y and fit must have same len")
	}
	if nCat < 1 {
		return nil, fmt.Errorf("nCat must be at least 1")
	}
	if trg == nil {
		return nil, fmt.Errorf("trg cannot be nil")
	}
	if len(y)%nCat != 0 {
		return nil, fmt.Errorf("len y not multiple of nCat")
	}
	for _, t := range trg {
		if t > nCat-1 {
			return nil, fmt.Errorf("trg index out of range")
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
			if y[ind+col] > 0.5 {
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
// Output: html plot file and/or plot in browser.
func KS(y []float64, fit []float64, nCat int, trg []int, logodds bool, plt *PlotDef, sl Slicer) (ks float64, notTarget *Desc, target *Desc, err error) {
	xy, err := Coalesce(y, fit, nCat, trg, logodds, sl)
	if err != nil {
		return -1.0, nil, nil, err
	}

	n := len(xy.X)
	// arrays to hold probabilities with observed 0's and 1's separately
	probNotTarget, probTarget := make([]float64, 0), make([]float64, 0)

	for row := 0; row < n; row++ {
		// append to appropriate slice
		switch {
		case xy.Y[row] > 0.5:
			probTarget = append(probTarget, xy.X[row])
		default:
			probNotTarget = append(probNotTarget, xy.X[row])
		}
	}
	notTarget, _ = NewDesc(nil, fmt.Sprintf("Value not in %v", trg))
	target, _ = NewDesc(nil, fmt.Sprintf("Value in %v", trg))
	notTarget.Populate(probNotTarget, false) // side-effect is probNotTarget is sorted
	target.Populate(probTarget, false)

	// Min & max of probabilities
	lower := math.Min(notTarget.Q[0], target.Q[0])
	upper := math.Max(notTarget.Q[len(notTarget.Q)-1], target.Q[len(target.Q)-1])

	// p is an array that is equally spaced between lower and upper
	p := make([]float64, 101)
	// these are cumulative distributions
	cumeNotTarget := make([]float64, 101)
	cumeTarget := make([]float64, 101)

	// for each value, find where it goes in our quantized slice p and increment corresponding element of cumeNotTarget
	for row := 0; row < len(probNotTarget); row++ {
		high := Min(int(100*(probNotTarget[row]-lower)/(upper-lower)), 100)
		cumeNotTarget[high]++
	}
	for row := 0; row < len(probTarget); row++ {
		high := Min(int(100*(probTarget[row]-lower)/(upper-lower)), 100)
		cumeTarget[high]++
	}

	//nFloat := float64(n)
	p[0], cumeNotTarget[0], cumeTarget[0] = lower, cumeNotTarget[0]/float64(len(probNotTarget)), cumeTarget[0]/float64(len(probTarget))
	ks, at := 0.0, 0.0
	// cumulate values and find KS
	for row := 1; row < 101; row++ {
		p[row] = lower + float64(row)*(upper-lower)/100.0
		cumeNotTarget[row] = cumeNotTarget[row]/float64(len(probNotTarget)) + cumeNotTarget[row-1]
		cumeTarget[row] = cumeTarget[row]/float64(len(probTarget)) + cumeTarget[row-1]
		if d := 100.0 * (cumeNotTarget[row] - cumeTarget[row]); d > ks {
			ks = d
			at = p[row]
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
		plt.XTitle = fmt.Sprintf("Probability value is in %v", trg)
		plt.YTitle = "Cumulative Distribution"
		lay := &grob.Layout{}
		lay.Legend = &grob.LayoutLegend{X: target.Q[0], Y: 1.0}
		err = Plotter(fig, lay, plt)
	}
	return
}

// Decile generates a decile plot of a softmax model that is reduced to a binary outcome.
//
//	y         observed multinomial values
//	fit       fitted softmax probabilities
//	trg       columns of y to be grouped into a single outcome. The complement is reduced to the alternate outcome.
//	logodds   if true, fit is in log odds space
//	plt       PlotDef plot options.  If plt is nil an error is generated.
//
// Output: html plot file and/or plot in browser.
func Decile(y []float64, fit []float64, nCat int, trg []int, logodds bool, plt *PlotDef, sl Slicer) error {
	if plt == nil {
		return fmt.Errorf("plt cannot be nil")
	}
	xy, err := Coalesce(y, fit, nCat, trg, logodds, sl)
	a, b := 0.0, 0.0
	for j := 0; j < xy.Len(); j++ {
		a += xy.X[j]
		b += xy.Y[j]
	}
	fmt.Println("obs, est avgs ", a/float64(xy.Len()), b/float64(xy.Len()))
	if err != nil {
		return err
	}
	if e := xy.Sort(); e != nil {
		return e
	}
	deciles, e := NewDesc([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, "fitted")
	if e != nil {
		return e
	}
	deciles.Populate(xy.X, false)
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
			return fmt.Errorf("decile group %d has no observations", g)
		}
		nFloat := float64(nDec[g])
		fDec[g] = fDec[g] / nFloat
		yDec[g] = yDec[g] / nFloat
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
		tr := &grob.Scatter{
			Type: grob.TraceTypeScatter,
			X:    []float64{fDec[g], fDec[g]},
			Y:    []float64{lower[g], upper[g]},
			Name: fmt.Sprintf("CI%d", g),
			Mode: grob.ScatterModeLines,
			Line: &grob.ScatterLine{Color: "black"},
		}
		fig.AddTraces(tr)
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
	plt.XTitle = fmt.Sprintf("Grouped mean probability value is in %v", trg)
	plt.YTitle = fmt.Sprintf("Grouped mean observed value is in %v", trg)
	plt.STitle = "95% CI assuming independence"
	err = Plotter(fig, &grob.Layout{}, plt)
	return err
}

func Assess(y []float64, fit []float64, nCat int, trg []int, logodds bool, cutoff float64, sl Slicer) (n int, precision float64, recall float64, accuracy float64, err error) {
	xy, err := Coalesce(y, fit, nCat, trg, logodds, sl)
	if err != nil {
		return
	}
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
		return 0, 0.0, 0.0, 0.0, fmt.Errorf("there are not positive outcomes")
	}
	precision = float64(correctYes) / float64(predTot) // fraction of Yes corrects to total predicted Yes
	recall = float64(correctYes) / float64(obsTot)     // fraction of Yes corrects to total actual Yes
	accuracy = float64(correct) / float64(len(xy.X))   // fraction of corrects
	n = len(xy.X)
	return
}
