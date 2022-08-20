package seafan

//TODO: add bias:true/false
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
	Output
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

//func (a *Activation) Check() bool {
//	if a == nil {
//		return false
//	}
//	return *a >= 0 && *a <= 3
//}

//type HidSize int

//func (s HidSize) Check() bool {
//	return s > 0 && s < 1000
//}

type FCLayer struct {
	Size     int
	Bias     bool
	Act      Activation
	ActParm  float64
	position int
}

// DOLayer specifies a dropout layer.  It occurs in the graph after dense layer AfterLayer (the input layer is layer 0).
type DOLayer struct {
	position int     // insert dropout after layer AfterLayer
	DropProb float64 // dropout probability
}

// ModSpec holds layers--each slice element is a layer
type ModSpec []string

// Args map holds layer arguments in key/val style
type Args map[string]string

// MakeArgs takes an argument string of the form "arg1:val1, arg2:val2, ...." and returns entries in key/val format
func MakeArgs(s string) (keyval Args, err error) {
	s = strings.ReplaceAll(s, " ", "")
	err = nil
	keyval = make(map[string]string)
	if !strings.Contains(s, ":") {
		return
	}
	entries := strings.Split(s, ",")
	for _, entry := range entries {
		if !strings.Contains(entry, ":") {
			err = fmt.Errorf("bad keyval: %s", entry)
			return
		}
		kv := strings.Split(entry, ":")
		if len(kv) != 2 {
			err = fmt.Errorf("bad keyval: %s", entry)
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
	return
}

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
			err = fmt.Errorf("illegal size")
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
	return
}

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
		return nil, fmt.Errorf("bad dropout probability <=0, >=1")
	}
	do := &DOLayer{DropProb: p}
	return do, nil
}

func (m ModSpec) Check() error {
	for _, ms := range m {
		l, _, e := Strip(ms)
		if e != nil {
			return e
		}
		if !strings.Contains(strings.ToLower(_Layer_name), l) {
			return fmt.Errorf("unknown layer: %s", l)
		}
	}
	return nil
}

func (m ModSpec) LType(i int) *Layer {
	if m.Check() != nil {
		return nil
	}
	if i < 0 || i >= len(m) {
		return nil
	}
	l, _, e := Strip(m[i])
	if e != nil {
		return nil
	}
	if i := strings.Index(strings.ToLower(_Layer_name), strings.ToLower(l)); i >= 0 {
		for ind, ix := range _Layer_index {
			if i == int(ix) {
				lay := Layer(ind)
				return &lay
			}
		}
	}
	return nil
}

func (m ModSpec) DropOut(loc int) *DOLayer {
	if *m.LType(loc) != DropOut {
		return nil
	}
	do, err := DropOutParse(m[loc])
	if err != nil {
		return nil
	}
	return do
}

func (m ModSpec) FC(loc int) *FCLayer {
	if *m.LType(loc) != FC {
		return nil
	}
	fc, err := FCParse(m[loc])
	if err != nil {
		return nil
	}
	return fc
}

func (m ModSpec) Inputs(p Pipeline) ([]*FType, error) {
	var err error
	modSpec := make([]*FType, 0)
	if *m.LType(0) != Input {
		return nil, fmt.Errorf("first layer is not Input")
	}
	_, inStr, e := Strip(m[0])
	if e != nil {
		return nil, e
	}

	var feat *FType
	fs := strings.Split(inStr, ",")
	for _, f := range fs {
		ft := f
		embCols := 0
		if strings.Contains(f, "E(") || strings.Contains(f, "e(") {
			l := strings.Split(ft, ",")
			if len(l) != 2 {
				return nil, fmt.Errorf("parse error")
			}
			ft = l[0][2:]
			var em int64
			em, err = strconv.ParseInt(l[1][0:len(l[1])-1], 10, 32)
			if err != nil {
				return nil, err
			}
			if em <= 1 {
				return nil, fmt.Errorf("embedding columns must be at least 2")
			}
			embCols = int(em)
		}
		feat = p.GetFType(ft)
		if feat == nil {
			return nil, fmt.Errorf("feature %s not found", f)
		}
		if feat.Role == FRCat {
			return nil, fmt.Errorf("feature %s is categorical--must convert to one-hot", feat.Name)
		}
		feat.EmbCols = embCols
		if embCols > 0 {
			if feat.Role != FROneHot {
				return nil, fmt.Errorf("feature %s must be one-hot", ft)
			}
			feat.Role = FREmbed
		}
		modSpec = append(modSpec, feat)
	}
	return modSpec, nil
}

func (m ModSpec) Output(p Pipeline) (*FType, error) {
	if *m.LType(len(m) - 1) != Output {
		return nil, nil
	}
	_, arg, e := Strip(m[len(m)-1])
	if e != nil {
		return nil, e
	}

	var feat *FType
	// target
	feat = p.GetFType(arg)
	if feat == nil {
		return nil, fmt.Errorf("feature %s not found", arg)
	}
	return feat, nil
}

func Strip(s string) (left, inner string, err error) {
	left, inner, err = "", "", nil
	s = strings.ToLower(strings.ReplaceAll(s, " ", ""))
	il := strings.Index(s, "(")
	if il <= 0 {
		return "", "", fmt.Errorf("bad (")
	}
	if s[len(s)-1:] != ")" {
		return "", "", fmt.Errorf("bad )")
	}
	left = s[0:il]
	inner = s[il+1 : len(s)-1]
	return
}

func (m ModSpec) Save(fileName string) (err error) {
	if err = m.Check(); err != nil {
		return
	}
	f, err := os.Create(fileName)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()
	for ind := 0; ind < len(m); ind++ {
		if _, err = f.WriteString(m[ind] + "\n"); err != nil {
			return
		}
	}
	return
}

func LoadModSpec(fileName string) (ms ModSpec, err error) {
	ms = make(ModSpec, 0)
	f, err := os.Open(fileName)
	if err != nil {
		return
	}
	defer func() { err = f.Close() }()

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
