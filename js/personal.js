import { app } from "../../scripts/app.js"

const alignKeys = {
  'ArrowUp': 'top',
  'ArrowDown': 'bottom',
  'ArrowLeft': 'left',
  'ArrowRight': 'right'
}

console.log("Loading personal extension")

app.registerExtension({
	name: 'Santi.Utils',

  _state: {
    selectedById: {},
    selectedLast: null,
    copiedValues: {},
    mousePos: [0, 0]
  },

	init() {
    const state = this._state
    const showSearchBoxW = app.canvas.onSearchBox
    const onSelectionChangeW = app.canvas.onSelectionChange

    app.canvas.onSearchBox = function(ev, query) {
      const results = showSearchBoxW?.apply(this, arguments) ?? []

      const re = new RegExp(query.toLowerCase().replaceAll(/\s+/g, '').split('').join('.*'))

      for (let name in LiteGraph.registered_node_types) {
        if (name.toLowerCase().replaceAll(/\s+/g, '').match(re)) {
          results.push(name)
        }
      }

      results.sort(function(a, b) {
        if (LiteGraph.registered_node_types[a].category === 'personal') {
          return -1
        } else if (LiteGraph.registered_node_types[b].category === 'personal') {
          return 1
        } else {
          return -1 // TODO least distance between letters?
        }
      })

      return results
    }

    app.canvas.onSelectionChange = function(nodesById) {
      if (isEmpty(nodesById)) {
        state.selectedById = {}
        state.selectedLast = null
      }

      let selectedNow = null

      for (let id in nodesById) {
        if (! (id in state.selectedById)) {
          selectedNow = nodesById[id]
          break
        }
      }

      state.selectedById = {...nodesById}
      state.selectedLast = selectedNow ?? state.selectedLast

      onSelectionChangeW?.apply(this, arguments)
    }

    const onMouseMove = (ev) => {
      this._state.mousePos = [ev.clientX, ev.clientY]
    }

		const onKeyDown = (ev) => {
      const state = this._state

      let wasHandled = true // set to `false` in the final `else` clause

      if (ev.shiftKey && ev.key in alignKeys && state.selectedLast != null) {
        this._alignNodes(app.canvas.selected_nodes, state.selectedLast, alignKeys[ev.key])

      } else if (ev.shiftKey && ev.key == 'L') {
        this._arrange(app.canvas.selected_nodes, state.selectedLast, 0, 40)
        this._alignNodes(app.canvas.selected_nodes, state.selectedLast, 'top')

      } else if (ev.shiftKey && ev.key == 'C') {
        this._arrange(app.canvas.selected_nodes, state.selectedLast, 1, 40)
        this._fit(app.canvas.selected_nodes, state.selectedLast, 0)
        this._alignNodes(app.canvas.selected_nodes, state.selectedLast, 'left')

      } else if (ev.shiftKey && ev.key == 'W') {
        this._fit(app.canvas.selected_nodes, state.selectedLast, 0)

      } else if (ev.shiftKey && ev.key == 'H') {
        this._fit(app.canvas.selected_nodes, state.selectedLast, 1)

      } else if (ev.shiftKey && ev.key == 'I') {
        console.log(state.selectedLast)

      } else if (ev.key == 'Tab') {
        // Fake mouse event (or showSearchBox fails):
        ev.clientX = state.mousePos ? state.mousePos[0] : 0
        ev.clientY = state.mousePos ? state.mousePos[1] : 0

        // Create end reposition dialog:
        const $canvas = app.canvas.canvas
        const $dialog = app.canvas.showSearchBox(ev, {hide_on_mouse_leave: false})

        const bounds = app.canvas.canvas.getBoundingClientRect()
        
        const width = 760;
        const left = bounds.left + ($canvas.clientWidth - width) / 2
        const top = bounds.top + ($canvas.clientHeight - $dialog.clientHeight) / 2

        $dialog.style = `width: ${width}px; left: ${left}px; top: ${top}px;`

        setTimeout(() => { $dialog.querySelector('input')?.focus() }, 50)

      } else if (ev.key == 'Escape') {
        document.querySelector('.graphdialog')?.close()

      } else {
        wasHandled = false
      }

      if (wasHandled) {
        // ev.preventDefault()
        // ev.stopPropagation()
      }
    }

		window.addEventListener("keydown", onKeyDown, true)
		window.addEventListener("mousemove", onMouseMove, true)
	},

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    const state = this._state
    const onCopyValues = (it, opt, ev, prev, node) => this._copyValues(node)
    const onPasteValues = (it, opt, ev, prev, node) => this._pasteValues()

    const getExtraMenuOptionsW = nodeType.prototype.getExtraMenuOptions
    const onNodeCreatedW = nodeType.prototype.onNodeCreated
    const onExecutedW = nodeType.prototype.onExecuted
    const onWidgetChangedW = nodeType.prototype.onWidgetChanged

    nodeType.prototype.getExtraMenuOptions = function(_, options) {
      getExtraMenuOptionsW?.apply(this, arguments)
      
      options.push(
        { content: "Copy values", callback: onCopyValues },
        { content: "Paste values", callback: onPasteValues }
      )
    }

    nodeType.prototype.onNodeCreated = function() {
      onNodeCreatedW?.apply(this, arguments)
      this.shape = "box"
      this.size[0] = 220
    }

    nodeType.prototype.onExecuted = function(ev) {
      onExecutedW?.apply(this, arguments)

     //  if (this.type == 'UseImage') {
        // this.outputs[0]["name"] = ev.text + " IMAGE"
      // this.outputs[0]["name"] = values[1] + " width"
     //  }
    }

    if (nodeType.comfyClass == 'GenerateImage') {
      nodeType.prototype.onWidgetChanged = function(name, value, oldValue, widget) {
        onWidgetChangedW?.apply(this, arguments)
        
        if (value == oldValue) {
          return
        }

        if (name == 'sampler') {
          switch(value) {
            case 'tcd':
              this.widgets[2].value = 8   // steps
              this.widgets[4].value = 1.0 // cfg
              this.widgets[5].value = 0.0 // pag
              this.widgets[6].value = 1.0 // denoise
              break

            case 'dpmpp_2m':
            case 'dpmpp_2m_sde':
            case 'deis':
              this.widgets[2].value = 35  // steps
              this.widgets[4].value = 4.0 // cfg
              this.widgets[5].value = 0.0 // pag
              this.widgets[6].value = 1.0 // denoise
              break

            case 'dpmpp_sde':
              this.widgets[2].value = 6 // steps
              this.widgets[4].value = 1.0 // cfg
              this.widgets[5].value = 0.0 // pag
              this.widgets[6].value = 1.0 // denoise
              break

            case 'euler':
              this.widgets[2].value = 24  // steps
              this.widgets[4].value = 1.0 // cfg
              this.widgets[5].value = 0.0 // pag
              this.widgets[6].value = 1.0 // denoise
              break

            case 'euler_ancestral':
              this.widgets[2].value = 24  // steps
              this.widgets[4].value = 6.0 // cfg
              this.widgets[5].value = 0.0 // pag
              this.widgets[6].value = 1.0 // denoise
          }
          
          app.graph.setDirtyCanvas(true)
        }
      }
    }
  },

  _alignNodes(nodes, anchor, alignTo) {
    LGraphCanvas.alignNodes(nodes, alignTo, anchor)
  },

  _arrange(nodes, anchor, axis, space) {
    const ns = Object.values(nodes) 

    if (!ns.includes(anchor)) ns.push(anchor) 
    if (ns.length < 2) return 

    ns.sort((a, b) => a.pos[axis] - b.pos[axis])

    // let nextPos = ns[0].pos[axis]

    let a = ns.findIndex(it => it === anchor)

    for (let i = a + 1; i < ns.length; i++) {
      const node = ns[i]
      const prev = ns[i - 1]

      if (axis == 0) {
        node.pos[axis] = prev.pos[axis] + (node.flags.collapsed ? node._collapsed_width : node.size[0]) + space
      } else {
        node.pos[axis] = prev.pos[axis] + (prev.flags.collapsed ? 0 : prev.size[1]) + space
      }
    }

    for (let i = a - 1; i >= 0; i--) {
      const node = ns[i]
      const next = ns[i + 1]

      if (axis == 0) {
        node.pos[axis] = next.pos[axis] - (node.flags.collapsed ? node._collapsed_width : node.size[0]) - space
      } else {
        node.pos[axis] = next.pos[axis] - (node.flags.collapsed ? 0 : node.size[1]) - space
      }
      console.log(node.pos)
    }
  },

  _fit(nodes, anchor, i) {
    for (let id in nodes) {
      nodes[id].size[i] = anchor.size[i]
    }
  },

  _copyValues(node) {
    const state = this._state
    
    const values = {} 
    node.widgets?.forEach((w, i) => values[w.name || i] = w.value)

    state.copiedValues = values
  },

  _pasteValues() {
    const state = this._state

    for (let id in state.selectedById) {
      const widgets = state.selectedById[id].widgets ?? []

      for (let widget of widgets) {
        if (widget.name in state.copiedValues) {
          widget.value = state.copiedValues[widget.name]
        }
      }
    }
  },
})

function isEmpty(obj) {
  for (let _ in obj) {
    return false
  }

  return true
}
