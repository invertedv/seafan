package seafan

import "fmt"

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
	q        []float64
}

func deDupe(xIn []float64) (xOut []float64) {
	xOut = append(xOut, xIn[0])
	for ind := 1; ind < len(xIn); ind++ {
		if xIn[ind] > xIn[ind-1] {
			xOut = append(xOut, xIn[ind])
		}
	}

	return xOut
}

// NewSlice makes a new Slice based on feat in Pipeline pipe.
// minCnt is the minimum # of obs a slice must have to be used.
// Restrict is a slice of values to restrict Iter to.
func NewSlice(feat string, minCnt int, pipe Pipeline, restrict []any) (*Slice, error) {
	d := pipe.Get(feat)

	if d == nil {
		return nil, Wrapper(ErrDiags, fmt.Sprintf("NewSlice: %s not found in pipeline", feat))
	}

	if d.FT.Role != FRCat && d.FT.Role != FRCts {
		return nil, Wrapper(ErrDiags, fmt.Sprintf("NewSlice: cannot slice type %v", d.FT.Role))
	}

	s := &Slice{feat: feat, minCnt: minCnt, pipe: pipe, index: -1, val: nil, data: d, restrict: restrict}

	if s.data.Summary.DistrC != nil {
		s.q = deDupe(s.data.Summary.DistrC.Q)
	}

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
			return s.data.Data.([]int32)[row] == s.index

		case FRCts:
			//q := deDupe(s.data.Summary.DistrC.Q)
			test := s.data.Data.([]float64)[row] >= s.q[s.index]
			switch s.index+1 == int32(len(s.q)-1) {
			case false:
				test = test && s.data.Data.([]float64)[row] < s.q[s.index+1]
			case true:
				test = test && s.data.Data.([]float64)[row] <= s.q[s.index+1]
			}
			return test
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
		if s.index+1 == int32(len(s.q)) {
			s.index = -1

			return false
		}

		// make title
		//		q := s.data.Summary.DistrC.Q
		qLab := make([]float64, len(s.q))
		copy(qLab, s.q)
		// if the feature is normalized, return it to original units for display
		if s.data.FT.Normalized {
			m := s.data.FT.FP.Location
			std := s.data.FT.FP.Scale

			for ind := 0; ind < len(qLab); ind++ {
				qLab[ind] = qLab[ind]*std + m
			}
		}

		s.title = fmt.Sprintf("%s between quantiles %v and %v", s.feat, qLab[s.index], qLab[s.index+1])
		s.val = qLab[s.index+1]

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
					s.title = fmt.Sprintf("field %s = %v", s.feat, s.val)

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
