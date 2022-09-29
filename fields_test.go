package seafan

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestFTypes_Save(t *testing.T) {
	//d := []any{"z", "a", "r", "a", "b"}
	d := []any{time.Date(2000, 11, 14, 10, 0, 0, 1, time.UTC),
		time.Date(2001, 12, 21, 0, 0, 1, 0, time.UTC)}
	lvl := ByPtr(NewRaw(d, nil))
	fp0 := &FParam{
		Location: 0,
		Scale:    0,
		Default:  d[0],
		Lvl:      lvl,
	}
	ft0 := &FType{
		Name:       "Field0",
		Role:       FRCat,
		Cats:       3,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         fp0,
	}

	fp1 := &FParam{
		Location: 2.0,
		Scale:    22.0,
		Default:  nil,
		Lvl:      nil,
	}
	ft1 := &FType{
		Name:       "Field1",
		Role:       FRCts,
		Cats:       0,
		EmbCols:    0,
		Normalized: true,
		From:       "",
		FP:         fp1,
	}

	fp2 := &FParam{
		Location: 0,
		Scale:    0,
		Default:  nil,
		Lvl:      nil,
	}
	ft2 := &FType{
		Name:       "Field2",
		Role:       FROneHot,
		Cats:       3,
		EmbCols:    0,
		Normalized: false,
		From:       "Field0",
		FP:         fp2,
	}

	fts := FTypes{ft0, ft1, ft2}
	fileName := os.TempDir() + "/seafanTest.json"
	e := fts.Save(fileName)
	assert.Nil(t, e)
	fts1, e := LoadFTypes(fileName)
	assert.Nil(t, e)

	for ind, ft := range fts {
		ft1 := fts1[ind]
		assert.Equal(t, ft1.Name, ft.Name)
		assert.Equal(t, ft1.Role, ft.Role)
		assert.Equal(t, ft1.Cats, ft.Cats)
		assert.Equal(t, ft1.Normalized, ft.Normalized)
		assert.Equal(t, ft1.From, ft.From)
		assert.Equal(t, ft1.FP.Location, ft.FP.Location)
		assert.Equal(t, ft1.FP.Scale, ft.FP.Scale)
		assert.Equal(t, ft1.FP.Default, ft.FP.Default)
		lvl1 := ft1.FP.Lvl
		lvl := ft.FP.Lvl
		if lvl == nil {
			assert.Equal(t, len(lvl1), 0)
			continue
		}
		assert.Equal(t, len(lvl1), len(lvl))
		for k, v := range lvl {
			assert.Equal(t, v, lvl1[k])
		}
	}
}

func TestFTypes_DropFields(t *testing.T) {
	d := []any{"z", "a", "r", "a", "b"}
	lvl := ByPtr(NewRaw(d, nil))
	fp0 := &FParam{
		Location: 0,
		Scale:    0,
		Default:  "z",
		Lvl:      lvl,
	}
	ft0 := &FType{
		Name:       "Field0",
		Role:       FRCat,
		Cats:       3,
		EmbCols:    0,
		Normalized: false,
		From:       "",
		FP:         fp0,
	}

	fp1 := &FParam{
		Location: 2.0,
		Scale:    22.0,
		Default:  nil,
		Lvl:      nil,
	}
	ft1 := &FType{
		Name:       "Field1",
		Role:       FRCts,
		Cats:       0,
		EmbCols:    0,
		Normalized: true,
		From:       "",
		FP:         fp1,
	}

	fp2 := &FParam{
		Location: 0,
		Scale:    0,
		Default:  nil,
		Lvl:      nil,
	}
	ft2 := &FType{
		Name:       "Field2",
		Role:       FROneHot,
		Cats:       3,
		EmbCols:    0,
		Normalized: false,
		From:       "Field0",
		FP:         fp2,
	}
	fts := FTypes{ft0, ft1, ft2}
	ftNew := fts.DropFields("Field1")
	ft := ftNew.Get("Field1")
	assert.Nil(t, ft)
	for _, ftName := range []string{"Field0", "Field2"} {
		ft := ftNew.Get(ftName)
		assert.NotNil(t, ft)
	}

}
