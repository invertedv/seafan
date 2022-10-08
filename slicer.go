package seafan

import "fmt"

const ctsCat = 4 // number of categories continuous fields are sliced into

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
		return nil, Wrapper(ErrDiags, fmt.Sprintf("NewSlice: %s not found in pipeline", feat))
	}

	if d.FT.Role != FRCat && d.FT.Role != FRCts {
		return nil, Wrapper(ErrDiags, fmt.Sprintf("NewSlice: cannot slice type %v", d.FT.Role))
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
			return s.data.Data.([]int32)[row] == s.index
		case FRCts:
			q := s.data.Summary.DistrC.Q

			switch s.index {
			case 0:
				return s.data.Data.([]float64)[row] < q[2] // under lower quartile
			case 1:
				return s.data.Data.([]float64)[row] >= q[2] && s.data.Data.([]float64)[row] < q[3] // lower quartile to median
			case 2:
				return s.data.Data.([]float64)[row] >= q[3] && s.data.Data.([]float64)[row] < q[4] // median to upper quartile
			case 3:
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
		if s.index == ctsCat {
			s.index = -1

			return false
		}
		s.val = fmt.Sprintf("Q%v", s.index+1)

		// make title
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
		case 1:
			s.title = fmt.Sprintf("%s Between Lower Quartile (%0.2f) and Median (%0.2f)", s.feat, qLab[2], qLab[3])
		case 2:
			s.title = fmt.Sprintf("%s Between Median (%0.2f) and Upper Quartile (%0.2f)", s.feat, qLab[3], qLab[4])
		case 3:
			s.title = fmt.Sprintf("%s Above Upper Quartile (%0.2f)", s.feat, qLab[4])
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
