package seafan

// structures/methods dealing with fields

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strconv"
)

// FType represents a single field. It holds key information about the feature: it's role, dimensions, summary info.
type FType struct {
	Name       string
	Role       FRole
	Cats       int
	EmbCols    int
	Normalized bool
	From       string
	FP         *FParam
}

type FTypes []*FType

// FParam -- field parameters -- is summary data about a field. These values may not be derived from the current
// data but are applied to the current data.
type FParam struct {
	Location float64 `json:"location"` // location parameter for *Cts
	Scale    float64 `json:"scale"`    // scale parameter for *Cts
	Default  any     `json:"default"`  // default level for *Dscrt
	Lvl      Levels  `json:"lvl"`      // map of values to int32 category for *Dscrt
}

// FRole is the role a feature plays
type FRole int

const (
	FRCts FRole = 0 + iota
	FRCat
	FROneHot
	FREmbed
)

//go:generate stringer -type=FRole

// Summary has descriptive statistics of a field using its current data.
type Summary struct {
	nRow   int    // size of the data
	DistrC *Desc  // summary of continuous field
	DistrD Levels // summary of discrete field
}

func (ft *FType) String() string {
	str := fmt.Sprintf("Field %s\n", ft.Name)
	switch ft.Role {
	case FRCts:
		str = fmt.Sprintf("%s\tcontinuous\n", str)
		if ft.Normalized {
			str = fmt.Sprintf("%s\tnormalized by:\n", str)
			str = fmt.Sprintf("%s\tlocation\t%.2f\n", str, ft.FP.Location)
			str = fmt.Sprintf("%s\tscale\t\t%.2f\n", str, ft.FP.Scale)
		}
	case FROneHot:
		str = fmt.Sprintf("%s\tone-hot\n", str)
		str = fmt.Sprintf("%s\tderived from feature %s\n", str, ft.From)
		str = fmt.Sprintf("%s\tlength %d\n", str, ft.Cats)
	case FREmbed:
		str = fmt.Sprintf("%s\tembedding\n", str)
		str = fmt.Sprintf("%s\tderived from feature %s\n", str, ft.From)
		str = fmt.Sprintf("%s\tlength %d\n", str, ft.Cats)
		str = fmt.Sprintf("%s\tembedding dimension of %d\n", str, ft.EmbCols)
	}
	return str
}

// Get returns the *FType of name
func (fts FTypes) Get(name string) *FType {
	for _, f := range fts {
		if f.Name == name {
			return f
		}
	}
	return nil
}

// fps is a json-friendly version of FParam
type fps struct {
	Location float64          `json:"location"` // location parameter for *Cts
	Scale    float64          `json:"scale"`    // scale parameter for *Cts
	Default  any              `json:"default"`  // default level for *Dscrt
	Kind     string           `json:"kind"`
	Lvl      map[string]int32 `json:"lvl"`
}

// ftype is a json-friendly version of FType
type fType struct {
	Name       string
	Role       FRole
	Cats       int
	EmbCols    int
	Normalized bool
	From       string
	FP         *fps
}

// Save saves FTypes to a json file--fileName
func (fts FTypes) Save(fileName string) (err error) {
	err = nil
	f, err := os.Create(fileName)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()
	out := make([]fType, 0)
	for _, ft := range fts {
		fpStr := &fps{}
		if ft.Role == FRCts || ft.Role == FRCat {
			var t reflect.Kind
			lvl := make(map[string]int32)
			for k, v := range ft.FP.Lvl {
				lvl[fmt.Sprintf("%v", k)] = v
				t = reflect.TypeOf(k).Kind()
			}
			fpStr = &fps{Location: ft.FP.Location, Scale: ft.FP.Scale, Default: ft.FP.Default}
			fpStr.Lvl = lvl
			fpStr.Kind = t.String()
		}
		ftype := fType{
			Name:       ft.Name,
			Role:       ft.Role,
			Cats:       ft.Cats,
			EmbCols:    ft.EmbCols,
			Normalized: ft.Normalized,
			From:       ft.From,
			FP:         fpStr,
		}
		out = append(out, ftype)
	}
	jfp, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return
	}
	if _, err = f.WriteString(string(jfp)); err != nil {
		return
	}
	return
}

// LoadFTypes loads a file created by the FTypes Save method
func LoadFTypes(fileName string) (fts FTypes, err error) {
	fts = nil
	err = nil
	f, err := os.Open(fileName)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()

	js, err := io.ReadAll(f)
	if err != nil {
		return
	}
	data := make([]fType, 0)
	if e := json.Unmarshal(js, &data); e != nil {
		fmt.Println(e)
		return nil, e
	}
	fts = make(FTypes, 0)
	for _, d := range data {
		ft := FType{
			Name:       d.Name,
			Role:       d.Role,
			Cats:       d.Cats,
			EmbCols:    d.EmbCols,
			Normalized: d.Normalized,
			From:       d.From,
			FP:         nil,
		}
		fp := FParam{Location: d.FP.Location, Scale: d.FP.Scale, Default: d.FP.Default}
		lvl := make(Levels)
		for k, v := range d.FP.Lvl {
			switch d.FP.Kind {
			case "string":
				lvl[k] = v
			case "int32":
				i, e := strconv.ParseInt(k, 10, 32)
				if e != nil {
					return nil, fmt.Errorf("cannot convert %s to int32", k)
				}
				lvl[int32(i)] = v
			case "int64":
				i, e := strconv.ParseInt(k, 10, 64)
				if e != nil {
					return nil, fmt.Errorf("cannot convert %s to int64", k)
				}
				lvl[i] = v
			}
		}
		fp.Lvl = lvl
		ft.FP = &fp
		fts = append(fts, &ft)
	}
	return
}
