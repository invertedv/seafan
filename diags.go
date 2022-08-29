package seafan

// diags.go implements model diagnostics

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"gonum.org/v1/gonum/stat"
	"math"
	"sort"
)

// Slicer is an optional function that returns true if the row is to be used in calculations. This is used to
// subset the diagnostics to specific values.
type Slicer func(row int) bool

// Slice implements generating Slicer functions for a feature.  These are used to slice through the values
// of a discrete feature. For continuous features, it slices by quartile.
type Slice struct {
	feat     string   // feature to slice
	minCnt   int      // a level of a feature must have at least minCnt obs to be used
	pipe     Pipeline // data pipeline
	index    int32    // current mapped level value we're working on
	val      any      // current actual value of a discrete feature
	title    string   // auto-generated title for plots
	data     *GDatum  // feat data
	restrict []any
}

// NewSlice makes a new Slice based on feat in Pipeline pipe.
// minCnt is the minimum # of obs a slice must have to be used.
// Restrict is a slice of values to restrict Iter to.
func NewSlice(feat string, minCnt int, pipe Pipeline, restrict []any) (*Slice, error) {
	d := pipe.Get(feat)

	if d == nil {
		return nil, fmt.Errorf("%s not found in pipeline", feat)
	}
	if d.FT.Role != FRCat && d.FT.Role != FRCts {
		return nil, fmt.Errorf("cannot slice type %v", d.FT.Role)
	}
	s := &Slice{feat: feat, minCnt: minCnt, pipe: pipe, index: -1, val: nil, data: d, restrict: restrict}
	return s, nil
}

// Title retrieves the auto-generated title
func (s *Slice) Title() string {
	return s.title
}

// Value returns the level of a discrete feature we're working on
func (s *Slice) Value() any {
	return s.val
}

// Index returns the mapped value of the current value
func (s *Slice) Index() int32 {
	return s.index
}

// SlicerAnd creates a Slicer that is s1 && s2
func SlicerAnd(s1, s2 Slicer) Slicer {
	return func(row int) bool {
		return s1(row) && s2(row)
	}
}

// SlicerOr creates a Slicer that is s1 || s2
func SlicerOr(s1, s2 Slicer) Slicer {
	return func(row int) bool {
		return s1(row) || s2(row)
	}
}

// MakeSlicer makes a Slicer function for the current value (discrete) or range (continuous) of the feature.
// Continuous features are sliced at the lower quartile, median and upper quartile, producing 4 slices.
func (s *Slice) MakeSlicer() Slicer {
	fx := func(row int) bool {
		switch s.data.FT.Role {
		case FRCat:
			s.title = fmt.Sprintf("field %s = %v", s.feat, s.val)
			return s.data.Data.([]int32)[row] == s.index
		case FRCts:
			q := s.data.Summary.DistrC.Q
			qLab := make([]float64, len(q))
			copy(qLab, q)
			// if the feature is normalized, return it to original units for display
			if s.data.FT.Normalized {
				m := s.data.FT.FP.Location
				std := s.data.FT.FP.Scale
				for ind := 0; ind < len(qLab); ind++ {
					qLab[ind] = qLab[ind]*std + m
				}
			}
			switch s.index {
			case 0:
				s.title = fmt.Sprintf("%s Less Than Lower Quartile (%0.2f)", s.feat, qLab[2])
				return s.data.Data.([]float64)[row] < q[2] // under lower quartile
			case 1:
				s.title = fmt.Sprintf("%s Between Lower Quartile (%0.2f) and Median (%0.2f)", s.feat, qLab[2], qLab[3])
				return s.data.Data.([]float64)[row] >= q[2] && s.data.Data.([]float64)[row] < q[3] // lower quartile to median
			case 2:
				s.title = fmt.Sprintf("%s Between Median (%0.2f) and Upper Quartile (%0.2f)", s.feat, qLab[3], qLab[4])
				return s.data.Data.([]float64)[row] >= q[3] && s.data.Data.([]float64)[row] < q[4] // median to upper quartile
			case 3:
				s.title = fmt.Sprintf("%s Above Upper Quartile (%0.2f)", s.feat, qLab[4])
				return s.data.Data.([]float64)[row] >= q[4] // above upper quartile
			}
		}
		return false
	}
	return fx
}

// Iter iterates through the levels (ranges) of the feature. Returns false when done.
func (s *Slice) Iter() bool {
	s.index++
	switch s.data.FT.Role {
	case FRCts:
		if s.index == 4 {
			s.index = -1
			return false
		}
		return true
	case FRCat:
		for {
			// find the level that corresponds to the mapped value s.index
			for k, v := range s.data.FT.FP.Lvl {
				if int(s.index) == s.data.FT.Cats {
					s.index = -1
					return false
				}
				if v == s.index {
					s.val = k
					// make sure it's in the current data set
					c, ok := s.data.Summary.DistrD[s.val]
					if !ok {
						s.index++
						continue
					}
					// and has enough data
					if int(c) <= s.minCnt {
						s.index++
						continue
					}
					if s.restrict == nil {
						return true
					}
					// check it's one of the values the user has restricted to
					for _, r := range s.restrict {
						if r == s.val {
							return true
						}
					}
					s.index++
				}
			}
		}
	}
	return false
}

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
// Target: html plot file and/or plot in browser.
func KS(xy *XY, plt *PlotDef) (ks float64, notTarget *Desc, target *Desc, err error) {

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
	notTarget, _ = NewDesc(nil, "not target") // fmt.Sprintf("Value not in %v", trg))
	target, _ = NewDesc(nil, "target")        // fmt.Sprintf("Value in %v", trg))
	notTarget.Populate(probNotTarget, false)  // side effect is probNotTarget is sorted
	target.Populate(probTarget, false)

	// Min & max of probabilities
	lower := math.Min(notTarget.Q[0], target.Q[0])
	upper := math.Max(notTarget.Q[len(notTarget.Q)-1], target.Q[len(target.Q)-1])

	// p is an array that is equally spaced between lower and upper
	p := make([]float64, 101)
	// these are cumulative distributions
	for k := 0; k < 101; k++ {
		p[k] = (float64(k)/100.0)*(upper-lower) + lower
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
	for k := 0; k < 101; k++ {
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
// Target: html plot file and/or plot in browser.
func Decile(xy *XY, plt *PlotDef) error {
	if plt == nil {
		return fmt.Errorf("plt cannot be nil")
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
func Assess(xy *XY, cutoff float64) (n int, precision float64, recall float64, accuracy float64, obs *Desc, fit *Desc, err error) {
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
		return 0, 0.0, 0.0, 0.0, nil, nil, fmt.Errorf("there are not positive outcomes")
	}
	precision = float64(correctYes) / float64(predTot) // fraction of Yes corrects to total predicted Yes
	recall = float64(correctYes) / float64(obsTot)     // fraction of Yes corrects to total actual Yes
	accuracy = float64(correct) / float64(len(xy.X))   // fraction of corrects
	n = len(xy.X)
	fit, err = NewDesc(nil, "fitted values")
	if err != nil {
		return
	}
	fit.Populate(xy.X, true)

	obs, err = NewDesc(nil, "observed values")
	if err != nil {
		return
	}
	obs.Populate(xy.Y, true)
	return
}
