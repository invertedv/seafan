package seafan

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"os"
	"testing"
)

func TestStrAct(t *testing.T) {
	inputs := []string{"garbage", "relu", "leakyrelu", "sigmoid", "LINEAR", "leakyrelu(0.5)"}
	expect := []any{nil, Relu, LeakyRelu, Sigmoid, Linear, LeakyRelu}
	parm := []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.5}
	for ind, i := range inputs {
		act, p := StrAct(i)
		if expect[ind] == nil {
			assert.Nil(t, act)
			continue
		}
		assert.Equal(t, *act, expect[ind].(Activation))
		assert.Equal(t, p, parm[ind])
	}
}

func TestFCParse(t *testing.T) {
	inputs := []string{
		"FC(size:2, activation:Relu)",
		"FC(size:3)",
		"FC(size:4, activation:LeakyRelu(-1))",
	}
	for ind, i := range inputs {
		_ = ind
		fc, e := FCParse(i)
		assert.Nil(t, e)
		fmt.Println(fc)
	}
}

func TestModSpec_FCs(t *testing.T) {
	mod := ModSpec{
		"Input(x1,x2,x3)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
	}
	fcs := mod.FCs()
	for _, f := range fcs {
		fmt.Println(f)
	}
	dos := mod.DropOuts()
	for _, d := range dos {
		fmt.Println(d)
	}
}

func TestStrip(t *testing.T) {
	inputs := []string{"ab(3)", "AB()", "r(as", "afdf)"}
	expectL := []string{"ab", "ab", "", ""}
	expectI := []string{"3", "", "", ""}
	for ind, i := range inputs {
		left, inner, _ := Strip(i)
		assert.Equal(t, left, expectL[ind])
		assert.Equal(t, expectI[ind], inner)
	}
}

func TestModSpec_Inputs(t *testing.T) {
	pipe := chPipe(100, "test1.csv")
	mod := ModSpec{
		"Input(x1,x2,x3)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
		"Output(ycts)",
	}
	expN := []string{"x1", "x2", "x3"}
	expR := []FRole{FRCts, FRCts, FRCts}
	fts, err := mod.Inputs(pipe)
	assert.Nil(t, err)
	for ind, ft := range fts {
		assert.Equal(t, expN[ind], ft.Name)
		assert.Equal(t, expR[ind], ft.Role)
	}
}

func TestModSpec_Output(t *testing.T) {
	pipe := chPipe(100, "test1.csv")
	mod := ModSpec{
		"Input(x1,x2,x3)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
		"Output(ycts)",
	}
	ft, e := mod.Output(pipe)
	assert.Nil(t, e)
	assert.Equal(t, ft.Name, "ycts")
	assert.Equal(t, ft.Role, FRCts)

	mod = ModSpec{
		"Input(x1,x2,x3)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
		"Output(yoh)",
	}
	ft, e = mod.Output(pipe)
	assert.Nil(t, e)
	assert.Equal(t, ft.Name, "yoh")
	assert.Equal(t, ft.Role, FROneHot)
}

func TestModSpec_Save(t *testing.T) {
	mod := ModSpec{
		"Input(x1,x2,x3)",
		"FC(size:3, activation:leakyrelu(0.1))",
		"Dropout(.1)",
		"FC(size:2)",
		"Dropout(.1)",
		"Output(ycts)",
	}
	outfile := os.TempDir() + "/testSave.txt"
	e := mod.Save(outfile)
	assert.Nil(t, e)
	mod1, e := LoadModSpec(outfile)
	assert.Nil(t, e)
	assert.ElementsMatch(t, mod, mod1)
}
