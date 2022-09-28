package seafan

// modspec.go handles specifying the model

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"reflect"
	"strconv"
	"strings"
)

// Layer types
type Layer int

const (
	Input Layer = 0 + iota
	FC
	DropOut
	Target
)

//go:generate stringer -type=Layer

// Activation types
type Activation int

const (
	Linear Activation = 0 + iota
	Relu
	LeakyRelu
	Sigmoid
	SoftMax
)

//go:generate stringer -type=Activation

// FCLayer has details of a fully connected layer
type FCLayer struct {
	Size    int
	Bias    bool
	Act     Activation
	ActParm float64
	//	position int
}

// DOLayer specifies a dropout layer.  It occurs in the graph after dense layer AfterLayer (the input layer is layer 0).
type DOLayer struct {
	//	position int     // insert dropout after layer AfterLayer
	DropProb float64 // dropout probability
}

// ModSpec holds layers--each slice element is a layer
type ModSpec []string

// Args map holds layer arguments in key/val style
type Args map[string]string

// MakeArgs takes an argument string of the form "arg1:val1, arg2:val2, ...." and returns entries in key/val format
func MakeArgs(s string) (keyval Args, err error) {
	s = strings.ReplaceAll(strings.ReplaceAll(s, " ", ""), "\n", "")
	err = nil
	keyval = make(map[string]string)

	if !strings.Contains(s, ":") {
		return
	}

	entries := strings.Split(s, ",")

	for _, entry := range entries {
		if !strings.Contains(entry, ":") {
			err = Wrapper(ErrModSpec, fmt.Sprintf("MakeArgs: bad keyval: %s", entry))

			return
		}

		kv := strings.Split(entry, ":")

		if len(kv) != 2 {
			err = Wrapper(ErrModSpec, fmt.Sprintf("MakeArgs: bad keyval: %s", entry))
		}

		keyval[kv[0]] = kv[1]
	}

	return
}

// Get returns a val from Args coercing to type kind.  Nil if fails.
func (kv Args) Get(key string, kind reflect.Kind) (val any) {
	val = nil
	valStr, ok := kv[key]

	if !ok {
		return nil
	}

	switch kind {
	case reflect.Float64:
		f, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			return
		}

		val = f
	case reflect.Int:
		i, err := strconv.ParseInt(valStr, 10, 32)
		if err != nil {
			return
		}

		val = int(i)
	case reflect.String:
		val = valStr
	case reflect.Bool:
		b, err := strconv.ParseBool(valStr)
		if err != nil {
			return
		}

		val = b
	}

	return val
}

// StrAct takes a string and returns corresponding Activation and any parameter.  Nil if fails.
func StrAct(s string) (*Activation, float64) {
	parm := float64(0)

	// takes a parameter?
	if strings.Contains(s, "(") {
		l, parmStr, e := Strip(s)
		if e != nil {
			return nil, 0.0
		}

		var err error
		parm, err = strconv.ParseFloat(parmStr, 64)

		if err != nil {
			return nil, 0.0
		}

		s = l
	}

	if i := strings.Index(strings.ToLower(_Activation_name), strings.ToLower(s)); i >= 0 {
		for ind, ix := range _Activation_index {
			if i == int(ix) {
				act := Activation(ind)

				return &act, parm
			}
		}
	}

	return nil, 0.0
}

// FCParse parses the arguments to an FC layer
func FCParse(s string) (fc *FCLayer, err error) {
	fc = nil
	kval, err := MakeArgs(strings.ToLower(s[3 : len(s)-1]))

	if err != nil {
		return
	}

	fc = &FCLayer{Act: Linear, Bias: true}

	if val := kval.Get("size", reflect.Int); val != nil {
		fc.Size = val.(int)
		if fc.Size < 1 {
			err = Wrapper(ErrModSpec, "FC: illegal size")

			return
		}
	}

	if val := kval.Get("activation", reflect.String); val != nil {
		if a, p := StrAct(val.(string)); a != nil {
			fc.Act = *a
			fc.ActParm = p
		}
	}

	if val := kval.Get("bias", reflect.Bool); val != nil {
		fc.Bias = val.(bool)
	}

	return fc, err
}

// DropOutParse parses the arguments to a drop out layer
func DropOutParse(s string) (*DOLayer, error) {
	_, args, err := Strip(s)
	if err != nil {
		return nil, err
	}

	p, err := strconv.ParseFloat(args, 64)

	if err != nil {
		return nil, err
	}

	if p <= 0.0 || p >= 1.0 {
		return nil, Wrapper(ErrModSpec, "DropOut: bad dropout probability <=0, >=1")
	}

	do := &DOLayer{DropProb: p}

	return do, nil
}

// Check checks that the layer name is valid
func (m ModSpec) Check() error {
	for _, ms := range m {
		l, _, e := Strip(ms)
		if e != nil {
			return e
		}

		if !strings.Contains(strings.ToLower(_Layer_name), strings.ToLower(l)) {
			return Wrapper(ErrModSpec, fmt.Sprintf("unknown layer: %s", l))
		}
	}

	return nil
}

// LType returns the layer type of layer i
func (m ModSpec) LType(i int) (*Layer, error) {
	if e := m.Check(); e != nil {
		return nil, e
	}

	if i < 0 || i >= len(m) {
		return nil, Wrapper(ErrModSpec, "layer name error")
	}

	l, _, e := Strip(m[i])

	if e != nil {
		return nil, e
	}

	if i := strings.Index(strings.ToLower(_Layer_name), strings.ToLower(l)); i >= 0 {
		for ind, ix := range _Layer_index {
			if i == int(ix) {
				lay := Layer(ind)

				return &lay, nil
			}
		}
	}

	return nil, Wrapper(ErrModSpec, "layer error")
}

// DropOut returns the *DoLayer for layer i, if it is of type DropOut.  Returns nil o.w.
func (m ModSpec) DropOut(loc int) *DOLayer {
	l, e := m.LType(loc)
	if e != nil {
		return nil
	}

	if *l != DropOut {
		return nil
	}

	do, err := DropOutParse(m[loc])

	if err != nil {
		return nil
	}

	return do
}

// FC returns the *FCLayer for layer i, if it is of type FC. Returns nil o.w.
func (m ModSpec) FC(loc int) *FCLayer {
	l, e := m.LType(loc)
	if e != nil {
		return nil
	}

	if *l != FC {
		return nil
	}

	fc, err := FCParse(m[loc])

	if err != nil {
		return nil
	}

	return fc
}

// Inputs returns the FTypes of the input features
func (m ModSpec) Inputs(p Pipeline) (FTypes, error) {
	var err error

	modSpec := make([]*FType, 0)
	l, e := m.LType(0)

	if e != nil {
		return nil, e
	}

	if *l != Input {
		return nil, Wrapper(ErrModSpec, "first layer is not Input")
	}

	_, inStr, e := Strip(m[0])

	if e != nil {
		return nil, e
	}

	var feat *FType

	fs := strings.Split(inStr, "+")

	for _, f := range fs {
		ft := f
		embCols := 0

		if strings.Contains(f, "E(") || strings.Contains(f, "e(") {
			l := strings.Split(ft, ",")
			if len(l) != 2 {
				return nil, Wrapper(ErrModSpec, "Inputs: parse error")
			}

			ft = l[0][2:]

			var em int64
			em, err = strconv.ParseInt(l[1][0:len(l[1])-1], 10, 32)

			if err != nil {
				return nil, err
			}

			if em <= 1 {
				return nil, Wrapper(ErrModSpec, "embedding columns must be at least 2")
			}

			embCols = int(em)
		}

		feat = p.GetFType(ft)

		if feat == nil {
			return nil, Wrapper(ErrModSpec, fmt.Sprintf("Inputs: feature %s not found", f))
		}

		if feat.Role == FRCat {
			return nil, Wrapper(ErrModSpec, fmt.Sprintf("feature %s is categorical--must convert to one-hot", feat.Name))
		}

		feat.EmbCols = embCols

		if embCols > 0 {
			if feat.Role != FROneHot && feat.Role != FREmbed {
				return nil, Wrapper(ErrModSpec, fmt.Sprintf("feature %s can't be continuous/categorical", ft))
			}

			feat.Role = FREmbed
		}

		modSpec = append(modSpec, feat)
	}

	return modSpec, nil
}

// Target returns the *FType of the target
func (m ModSpec) Target(p Pipeline) (*FType, error) {
	l, e := m.LType(len(m) - 1)
	if e != nil {
		return nil, e
	}

	if *l != Target {
		return nil, nil
	}

	_, arg, e := Strip(m[len(m)-1])

	if e != nil {
		return nil, e
	}

	feat := p.GetFType(arg)
	if feat == nil {
		return nil, Wrapper(ErrModSpec, fmt.Sprintf("feature %s not found", arg))
	}

	return feat, nil
}

// Save ModSpec
func (m ModSpec) Save(fileName string) (err error) {
	if err = m.Check(); err != nil {
		return
	}

	f, err := os.Create(fileName)

	if err != nil {
		return
	}

	defer func() { _ = f.Close() }()

	for ind := 0; ind < len(m); ind++ {
		line := strings.ReplaceAll(strings.ReplaceAll(m[ind], " ", ""), "\n", "")
		if _, err = f.WriteString(line + "\n"); err != nil {
			return
		}
	}

	return
}

// LoadModSpec loads a ModSpec from file
func LoadModSpec(fileName string) (ms ModSpec, err error) {
	ms = make(ModSpec, 0)
	f, err := os.Open(fileName)

	if err != nil {
		return
	}

	defer func() { _ = f.Close() }()

	buf := bufio.NewReader(f)

	for err != io.EOF {
		var l []byte
		l, err = buf.ReadBytes('\n')

		if err != nil && err != io.EOF {
			return nil, err
		}

		line := strings.ReplaceAll(string(l), "\n", "")

		if line != "" {
			ms = append(ms, line)
		}
	}

	err = ms.Check()

	return
}

// Strip is a utility that takes a string of the form "Func(args)" and returns "Func" and "args"
func Strip(s string) (left, inner string, err error) {
	left, inner, err = "", "", nil

	s = strings.ReplaceAll(strings.ReplaceAll(s, " ", ""), "\n", "")
	il := strings.Index(s, "(")

	if il <= 0 {
		return "", "", Wrapper(ErrModSpec, "bad (")
	}

	if s[len(s)-1:] != ")" {
		return "", "", Wrapper(ErrModSpec, "bad )")
	}

	left = s[0:il]
	inner = s[il+1 : len(s)-1]

	return
}
