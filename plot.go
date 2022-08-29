package seafan

// plot.go implements routines to make simple plotly plots easy.

import (
	"fmt"
	grob "github.com/MetalBlueberry/go-plotly/graph_objects"
	"github.com/MetalBlueberry/go-plotly/offline"
	"math/rand"
	"os"
	"os/exec"
	"time"
)

// PlotDef specifies Plotly Layout features I commonly use.
type PlotDef struct {
	Title    string  // Title - plot title
	XTitle   string  // XTitle - x-axis title
	YTitle   string  // Ytitle - y-axis title
	STitle   string  // STitle - sub-title (under the x-axis)
	Legend   bool    // Legend - true = show legend
	Height   float64 // Height - height of graph, in pixels
	Width    float64 // Width - width of graph, in pixels
	Show     bool    // Show - true = show graph in browser
	FileName string  // FileName - output file for graph (in html)
}

// Plotter plots the Plotly Figure fig with Layout lay.  The layout is augmented by
// features I commonly use.
//
//	fig      plotly figure
//	lay      plotly layout (nil is OK)
//	pd       PlotDef structure with plot options.
//
// lay can be initialized with any additional layout options needed.
func Plotter(fig *grob.Fig, lay *grob.Layout, pd *PlotDef) error {
	if lay == nil {
		lay = &grob.Layout{}
	}
	lay.Title = &grob.LayoutTitle{Text: pd.Title}
	lay.Yaxis = &grob.LayoutYaxis{Title: &grob.LayoutYaxisTitle{Text: pd.YTitle}}
	lay.Yaxis.Showline = grob.True // oddly this is false by default
	xTitle := pd.XTitle

	if pd.STitle != "" {
		xTitle += fmt.Sprintf("<br>%s", pd.STitle)
	}
	lay.Xaxis = &grob.LayoutXaxis{Title: &grob.LayoutXaxisTitle{Text: xTitle}}
	if !pd.Legend {
		lay.Showlegend = grob.False
	}
	if pd.Width > 0.0 {
		lay.Width = pd.Width
	}
	if pd.Height > 0.0 {
		lay.Height = pd.Height
	}
	fig.Layout = lay
	if pd.FileName != "" {
		offline.ToHtml(fig, pd.FileName)
	}
	if pd.Show {
		tmp := false
		if pd.FileName == "" {
			tmp = true
			// create temp file.  We'll return this, in case it's needed
			rand.Seed(time.Now().UnixMicro())
			pd.FileName = fmt.Sprintf("%s/plotly%d.html", os.TempDir(), rand.Uint32())
		}
		offline.ToHtml(fig, pd.FileName)
		cmd := exec.Command(Browser, "-url", pd.FileName)
		if e := cmd.Start(); e != nil {
			return e
		}
		if tmp {
			// need to pause while browser loads graph
			time.Sleep(time.Second)
			if e := os.Remove(pd.FileName); e != nil {
				return e
			}
		}
	}
	return nil
}
