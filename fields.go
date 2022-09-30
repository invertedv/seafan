package seafan

// fields.go implements structures/methods dealing with fields

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"strconv"
	"time"
)

// FType represents a single field. It holds key information about the feature: its role, dimensions, summary info.
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

// DropFields will drop fields from the FTypes
func (fts FTypes) DropFields(dropFields ...string) FTypes {
	ftOut := make(FTypes, 0)
	for _, ft := range fts {
		keep := true
		for _, nm := range dropFields {
			if ft.Name == nm {
				keep = false
				break
			}
		}
		if keep {
			ftOut = append(ftOut, ft)
		}
	}
	return ftOut
}

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
	NRows  int    // size of the data
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

	defer func() { _ = f.Close() }()

	out := make([]fType, 0)

	for _, ft := range fts {
		fpStr := &fps{}

		if ft.Role == FRCts || ft.Role == FRCat {
			var dataType string

			lvl := make(map[string]int32)

			for k, v := range ft.FP.Lvl {
				dataType = reflect.TypeOf(k).Kind().String()
				kOut := fmt.Sprintf("%v", k)
				if dataType == "struct" {
					val, ok := k.(time.Time)
					if !ok {
						return Wrapper(ErrFields, fmt.Sprintf("(FTypes) Save: unexpect struct type, field %s", ft.Name))
					}
					dataType = "date"
					kOut = val.Format(time.RFC3339)
				}
				lvl[kOut] = v
			}

			fpStr = &fps{Location: ft.FP.Location, Scale: ft.FP.Scale, Default: ft.FP.Default}
			fpStr.Lvl = lvl
			fpStr.Kind = dataType
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

	return err
}

// LoadFTypes loads a file created by the FTypes Save method
func LoadFTypes(fileName string) (fts FTypes, err error) {
	fts = nil
	err = nil
	f, err := os.Open(fileName)

	if err != nil {
		return
	}

	defer func() { _ = f.Close() }()

	js, err := io.ReadAll(f)
	if err != nil {
		return
	}

	data := make([]fType, 0)

	if e := json.Unmarshal(js, &data); e != nil {
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

		switch d.FP.Kind {
		case "string":
			fp.Default = fmt.Sprintf("%v", d.FP.Default)
		case "int32":
			val := int32(d.FP.Default.(float64))
			fp.Default = val
		case "int64":
			val := int64(d.FP.Default.(float64))
			fp.Default = val
		case "date":
			val, e := time.Parse(time.RFC3339, fmt.Sprintf("%s", d.FP.Default))
			if e != nil {
				return nil, Wrapper(ErrFields, fmt.Sprintf("LoadTypes: cannot convert default value %v to date", d.FP.Default))
			}
			fp.Default = val
		}

		lvl := make(Levels)
		for k, v := range d.FP.Lvl {
			switch d.FP.Kind {
			case "string":
				lvl[k] = v
			case "int32":
				i, e := strconv.ParseInt(k, 10, 32)
				if e != nil {
					return nil, Wrapper(ErrFields, fmt.Sprintf("LoadFTypes: cannot convert %s to int32", k))
				}

				lvl[int32(i)] = v
			case "int64":
				i, e := strconv.ParseInt(k, 10, 64)
				if e != nil {
					return nil, Wrapper(ErrFields, fmt.Sprintf("LoadFTypes: cannot convert %s to int64", k))
				}

				lvl[i] = v
			case "date":
				dt, e := time.Parse(time.RFC3339, k)
				if e != nil {
					return nil, Wrapper(ErrFields, fmt.Sprintf("LoadFTypes: cannot convert %s to date", k))
				}
				lvl[dt] = v
			}
		}

		fp.Lvl = lvl
		ft.FP = &fp
		fts = append(fts, &ft)
	}
	return fts, err
}
