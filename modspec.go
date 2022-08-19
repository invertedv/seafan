package seafan

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

func (a *Activation) Check() bool {
	if a == nil {
		return false
	}
	return *a >= 0 && *a <= 3
}

type HidSize int

func (s HidSize) Check() bool {
	return s > 0 && s < 1000
}

type FCLayer struct {
	Size     HidSize
	Act      Activation
	ActParm  float64
	position int
}

// DOLayer specifies a dropout layer.  It occurs in the graph after dense layer AfterLayer (the input layer is layer 0).
type DOLayer struct {
	AfterLayer int     // insert dropout after layer AfterLayer
	DropProb   float64 // dropout probability
}

type ModSpec []string

type Args map[string]string

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
	}
	return
}

func FCParse(s string) (fc *FCLayer, err error) {
	fc = nil
	kval, err := MakeArgs(strings.ToLower(s[3 : len(s)-1]))
	if err != nil {
		return
	}
	fc = &FCLayer{Act: Linear}
	if val := kval.Get("size", reflect.Int); val != nil {
		fc.Size = HidSize(val.(int))
		if !fc.Size.Check() {
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
	return
}

func (m ModSpec) FC(l int) *FCLayer {
	for _, f := range m.FCs() {
		if f.position == l {
			return f
		}
	}
	return nil
}

func (m ModSpec) FCs() []*FCLayer {
	if e := m.Check(); e != nil {
		return nil
	}
	fcs := make([]*FCLayer, 0)
	position := 0
	for ind, term := range m {
		l := m.LType(ind)
		if l == nil {
			return nil
		}
		if *l != FC {
			continue
		}
		fc, err := FCParse(term)
		if err != nil {
			return nil
		}
		fc.position = position
		position++
		fcs = append(fcs, fc)
	}
	return fcs
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

func (m ModSpec) DropOut(after int) *DOLayer {
	for _, do := range m.DropOuts() {
		if do.AfterLayer == after {
			return do
		}
	}
	return nil
}

func (m ModSpec) DropOuts() []*DOLayer {
	if e := m.Check(); e != nil {
		return nil
	}
	dos := make([]*DOLayer, 0)
	position := 0
	after := -1
	for ind, term := range m {
		l := m.LType(ind)
		if l == nil {
			return nil
		}
		if *l == FC {
			after++
		}
		if *l != DropOut {
			continue
		}
		do, err := DropOutParse(term)
		if err != nil {
			return nil
		}
		do.AfterLayer = after
		position++
		dos = append(dos, do)
	}
	return dos
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
		return nil, fmt.Errorf("last entry is not output")
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
