window.ilm = window.ilm || {};

(function(ilm) {
  ilm.config = {};

  ilm.config.OPERATE_UNDO_STACK_MAX_LENGTH = 128;
  ilm.config.DEFAULT_TEXT_SEED = 0;
  ilm.config.SERVER_ADDRESS = null;
  ilm.config.SERVER_CONNECT_ISSUE_MSG =
    "Could not connect to backend. Please make sure your server address looks something like '123a4bcd.ngrok.io'. If it does, please try pressing 'Connect' after a few seconds. If that doesn't work, please rerun the second cell on the Colab notebook and try again. If the demo still doesn't work, we sincerely apologize and will hopefully fix it ASAP.";
})(window.ilm);

window.ilm = window.ilm || {};

(function(ilm) {
  ilm.api = {};
  ilm.api.serverAddress = "";

  function ilmUrl(api_method) {
    return `${location.protocol}//${ilm.api.serverAddress}/${api_method}`;
  }

  ilm.api.fetch = async function(api_method, arg_dict) {
    return fetch(ilmUrl(api_method), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(arg_dict)
    })
      .then(async response => {
        if (response.ok) {
          return await response.text();
        } else {
          throw new Error(await response.text());
        }
      })
      .catch(error => {
        // Ugly hack. Ngrok throttles requests with 429.
        if (String(error).includes("TypeError")) {
          error = "Too many requests. Please try again in a minute.";
        }
        throw new Error(error);
      });
  };

  ilm.api.fetchJson = async function(api_method, arg_dict) {
    return fetch(ilmUrl(api_method), {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(arg_dict)
    })
      .then(async response => {
        if (response.ok) {
          return await response.json();
        } else {
          throw new Error(await response.text());
        }
      })
      .catch(error => {
        // Ugly hack. Ngrok throttles requests with 429.
        if (String(error).includes("TypeError")) {
          error = "Too many requests. Please try again in a minute.";
        }
        throw new Error(error);
      });
  };
})(window.ilm);

(function(ilm) {
  ilm.helpers = {};

  ilm.helpers.createDivWithClassName = function(className) {
    const div = document.createElement("div");
    div.setAttribute("class", className);
    return div;
  };

  ilm.helpers.uuidv4 = function() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(c) {
      var r = (Math.random() * 16) | 0,
        v = c == "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  };
})(window.ilm);

(function(ilm) {
  // DOM elements
  let operateDivEl;
  let operateRandomButtonEl;
  let operateEditButtonEl;
  let operateClozeButtonEl;
  let operateUndoButtonEl;
  let operateRedoButtonEl;
  let operateUiDivEl;
  const operateCoordToElement = new Map();
  let editDivEl;
  let editTextAreaEl;
  let editDoneButtonEl;

  // State
  const operateCoordsCloze = new Set();
  let operateXmlCurrent;
  const operateUndoStack = new Array();
  const operateRedoStack = new Array();
  let operateLastClozedXml = null;
  let operateLastClozedCoords = null;

  let sessionId;

  function undoStackClear() {
    operateUndoButtonEl.disabled = true;
    operateUndoStack.length = 0;
    operateRedoButtonEl.disabled = true;
    operateRedoStack.length = 0;
  }

  function undoStackPushCurrent(clearRedo) {
    if (operateXmlCurrent !== undefined) {
      while (
        operateUndoStack.length >= ilm.config.OPERATE_UNDO_STACK_MAX_LENGTH
      ) {
        operateUndoStack.shift();
      }
      operateUndoStack.push(operateXmlCurrent);
      operateUndoButtonEl.disabled = false;
      if (clearRedo === undefined || clearRedo === true) {
        operateRedoStack.length = 0;
        operateRedoButtonEl.disabled = true;
      }
    }
  }

  async function operateEditClick() {
    const result = await ilm.api.fetch("flatten", {
      document_xml: operateXmlCurrent
    });
    editView(result);
    operateLastClozedXml = null;
  }

  async function editDoneClick() {
    const result = await ilm.api.fetch("parse", {
      document_raw: editTextAreaEl.value
    });

    document.getElementById("edit").hidden = true;
    document.getElementById("operate").hidden = false;
    document.getElementById("history-controls").hidden = false;

    undoStackPushCurrent();
    operateDrawXml(result);
  }

  function editView(text) {
    document.getElementById("edit").hidden = false;
    document.getElementById("operate").hidden = true;
    document.getElementById("history-controls").hidden = true;

    editTextAreaEl.value = text;
  }

  function indicateClozeCallbackFactory(xmlEl, htmlEl) {
    let coord = xmlEl.getAttribute("value");
    operateCoordToElement.set(coord, htmlEl);
    const coordSplit = coord.split(",");
    if (coordSplit.length === 3) {
      coord += ",1";
    }

    return function(event) {
      if (operateCoordsCloze.has(coord)) {
        htmlEl.classList.remove("selected");
        operateCoordsCloze.delete(coord);
      } else {
        htmlEl.classList.add("selected");
        operateCoordsCloze.add(coord);
      }

      event.stopPropagation();
    };
  }

  function operateDrawXml(xml) {
    operateCoordToElement.clear();
    operateCoordsCloze.clear();

    operateXmlCurrent = xml;
    const xmlRoot = new DOMParser()
      .parseFromString(xml, "text/xml")
      .getElementsByTagName("ilm-document")[0];

    const xmlDoc = xmlRoot.getElementsByTagName("ilm-paragraph");
    const htmlDoc = ilm.helpers.createDivWithClassName("ilm-document");
    htmlDoc.onmousedown = indicateClozeCallbackFactory(xmlRoot, htmlDoc);
    htmlDoc.onmouseenter = function() {
      $(this).addClass("hovered");
    };
    htmlDoc.onmouseleave = function() {
      $(this).removeClass("hovered");
    };

    for (let i = 0; i < xmlDoc.length; ++i) {
      const xmlParagraph = xmlDoc[i].getElementsByTagName("ilm-sentence");
      const htmlParagraph = ilm.helpers.createDivWithClassName("ilm-paragraph");
      htmlParagraph.onmousedown = indicateClozeCallbackFactory(
        xmlDoc[i],
        htmlParagraph
      );

      for (let j = 0; j < xmlParagraph.length; ++j) {
        const xmlSentence = xmlParagraph[j].getElementsByTagName("ilm-word");
        const htmlSentence = ilm.helpers.createDivWithClassName("ilm-sentence");
        htmlSentence.onmousedown = indicateClozeCallbackFactory(
          xmlParagraph[j],
          htmlSentence
        );

        for (let k = 0; k < xmlSentence.length; ++k) {
          const xmlWord = xmlSentence[k];
          const htmlWord = ilm.helpers.createDivWithClassName("ilm-word");
          htmlWord.innerHTML = xmlWord.innerHTML;
          htmlWord.onmousedown = indicateClozeCallbackFactory(
            xmlWord,
            htmlWord
          );
          htmlWord.onmouseenter = function() {
            $(this).addClass("hovered");
            $(this)
              .parent()
              .removeClass("hovered");
          };
          htmlWord.onmouseleave = function() {
            $(this).removeClass("hovered");
            $(this)
              .parent()
              .addClass("hovered");
          };
          htmlSentence.appendChild(htmlWord);
        }

        htmlSentence.onmouseenter = function() {
          $(this).addClass("hovered");
          $(this)
            .parent()
            .removeClass("hovered");
        };
        htmlSentence.onmouseleave = function() {
          $(this).removeClass("hovered");
          $(this)
            .parent()
            .addClass("hovered");
        };
        htmlParagraph.appendChild(htmlSentence);
      }
      htmlParagraph.onmouseenter = function() {
        $(this).addClass("hovered");
        $(this)
          .parent()
          .removeClass("hovered");
      };
      htmlParagraph.onmouseleave = function() {
        $(this).removeClass("hovered");
        $(this)
          .parent()
          .addClass("hovered");
      };
      htmlDoc.appendChild(htmlParagraph);
    }

    while (operateUiDivEl.firstChild) {
      operateUiDivEl.removeChild(operateUiDivEl.firstChild);
    }
    operateUiDivEl.appendChild(htmlDoc);
  }

  async function operateRandomClick(seed) {
    if (typeof seed !== "number") {
      seed = null;
    }

    const result = (await ilm.api.fetchJson("random_documents", {
      num_examples: 1,
      as_xml: true,
      seed: seed
    }))[0];
    undoStackPushCurrent();
    operateLastClozedXml = null;
    operateDrawXml(result);
  }

  async function modelSwitchChange() {
    await operateRandomClick(ilm.config.DEFAULT_TEXT_SEED);
    undoStackClear();
    operateLastClozedXml = null;
    sessionId = ilm.helpers.uuidv4();
  }

  function parseMaskCoords(coords) {
    const ngramsMerged = new Set();
    coords.forEach(coord => {
      const coordSplit = coord.split(",");
      if (coordSplit.length === 4) {
        ngramsMerged.add(coord);
      }
    });
    ngramsMerged.forEach(coord => {
      coords.delete(coord);
    });

    while (true) {
      if (ngramsMerged.size === 0) {
        break;
      }

      const coordToNgramLength = new Map();
      ngramsMerged.forEach(ncoord => {
        const ncoordSplit = ncoord.split(",");
        const coordSplit = ncoordSplit.slice(0, 3);
        const n = Number(ncoordSplit[3]);
        coordToNgramLength.set(coordSplit.join(","), n);
      });

      let merged = false;
      const ngramsArr = Array.from(ngramsMerged);
      for (let i = 0; i < ngramsArr.length; ++i) {
        const ncoord = ngramsArr[i];
        const ncoordSplit = ncoord.split(",");
        for (let j = 0; j < 4; ++j) {
          ncoordSplit[j] = Number(ncoordSplit[j]);
        }

        const nextWordCoordSplit = ncoordSplit.slice(0, 3);
        nextWordCoordSplit[2] += ncoordSplit[3];
        const nextWordCoord = nextWordCoordSplit.join(",");
        if (coordToNgramLength.has(nextWordCoord)) {
          const n = ncoordSplit[3];
          const nextWordN = coordToNgramLength.get(nextWordCoord);
          const nextWordNcoord = nextWordCoordSplit;
          nextWordNcoord.push(nextWordN);
          ngramsMerged.delete(ncoord);
          ngramsMerged.delete(nextWordNcoord.join(","));
          ncoordSplit[3] += nextWordN;
          ngramsMerged.add(ncoordSplit.join(","));
          merged = true;
          break;
        }
      }

      if (!merged) {
        break;
      }
    }

    ngramsMerged.forEach(function(coord) {
      coords.add(coord);
    });

    const cleanedCoords = new Set();

    coords.forEach(function(coord) {
      if (coord.length === 7 && coord[6] == "1") {
        coord = coord.slice(0, 5);
      }
      cleanedCoords.add(coord);
    });

    return cleanedCoords;
  }

  async function operateClozeClick() {
    const elementsToClearCoords = new Set(operateCoordsCloze);

    let xmlToUse = operateXmlCurrent;
    let coordsToUse = operateCoordsCloze;

    if (operateCoordsCloze.size === 0) {
      if (operateLastClozedXml !== null) {
        xmlToUse = operateLastClozedXml;
        coordsToUse = new Set(operateLastClozedCoords);
      } else {
        alert("Please select text to infill by clicking buttons!");
        return;
      }
    }

    operateLastClozedXml = xmlToUse;
    operateLastClozedCoords = new Set(coordsToUse);

    coordsToUse = parseMaskCoords(coordsToUse);

    const requestAttrs = {
      session_id: sessionId,
      document_xml: xmlToUse,
      document_mask_coords: Array.from(coordsToUse)
    };

    requestAttrs.temperature = Number(
      document.getElementById("temperature").value
    );
    requestAttrs.topk = Number(document.getElementById("topk").value);
    if (requestAttrs.topk <= 0) {
      delete requestAttrs.topk;
    }
    requestAttrs.nucleus = Number(document.getElementById("nucleus").value);

    let result;
    try {
      document.getElementById("infill-button-text").hidden = true;
      document.getElementById("infill-button-loading").hidden = false;
      result = await ilm.api.fetch("cloze", requestAttrs);
    } catch (error) {
      alert(error);
    } finally {
      document.getElementById("infill-button-text").hidden = false;
      document.getElementById("infill-button-loading").hidden = true;
    }

    elementsToClearCoords.forEach(function(coord) {
      const coordSplit = coord.split(",");
      if (coordSplit.length == 4) {
        coord = coordSplit.slice(0, 3).join(",");
      }
      const el = operateCoordToElement.get(coord);
      el.style.backgroundColor = "";
    });
    operateCoordsCloze.clear();

    if (result !== undefined) {
      undoStackPushCurrent();
      operateDrawXml(result);
    }
  }

  function operateUndoClick() {
    if (operateUndoStack.length > 0) {
      if (operateXmlCurrent !== undefined) {
        operateRedoStack.push(operateXmlCurrent);
        operateRedoButtonEl.disabled = false;
      }
      operateDrawXml(operateUndoStack.pop());
      if (operateUndoStack.length === 0) {
        operateUndoButtonEl.disabled = true;
      }
    }
    operateLastClozedXml = null;
  }

  function operateRedoClick() {
    if (operateRedoStack.length > 0) {
      undoStackPushCurrent(false);
      operateDrawXml(operateRedoStack.pop());
      if (operateRedoStack.length === 0) {
        operateRedoButtonEl.disabled = true;
      }
    }
    operateLastClozedXml = null;
  }

  function onKeyDown(event) {
    if (document.getElementById("frontend").hidden === true) {
      return;
    } else if (document.getElementById("operate").hidden === true) {
      return;
    } else if (event.repeat) {
      return;
    } else if (event.key === " ") {
      event.preventDefault();
      operateClozeClick();
    }
  }

  async function connect(serverAddress) {
    ilm.api.serverAddress = serverAddress;
    try {
      const result = await ilm.api.fetchJson("random_documents", {});
    } catch {
      alert(ilm.config.SERVER_CONNECT_ISSUE_MSG);
      return;
    }

    document.getElementById("frontend").hidden = false;
    document.getElementById("server-instructions").hidden = true;
    document.getElementById("server-connect").innerHTML = "Reconnect";
    await modelSwitchChange();
  }

  async function initUi() {
    operateDivEl = document.getElementById("operate");
    operateRandomButtonEl = document.getElementById("random-button");
    operateEditButtonEl = document.getElementById("edit-button");
    operateClozeButtonEl = document.getElementById("infill-button");
    operateUndoButtonEl = document.getElementById("undo-button");
    operateRedoButtonEl = document.getElementById("redo-button");
    operateUiDivEl = document.getElementById("operate-ui");
    editDivEl = document.getElementById("edit");
    editTextAreaEl = document.getElementById("edit-text");
    editDoneButtonEl = document.getElementById("done-button");

    document.addEventListener("keydown", onKeyDown);
    operateRandomButtonEl.onclick = operateRandomClick;
    operateEditButtonEl.onclick = operateEditClick;
    operateClozeButtonEl.onclick = operateClozeClick;
    operateUndoButtonEl.onclick = operateUndoClick;
    operateRedoButtonEl.onclick = operateRedoClick;
    editDoneButtonEl.onclick = editDoneClick;

    sessionId = ilm.helpers.uuidv4();

    const serverConnectButton = document.getElementById("server-connect");
    serverConnectButton.onclick = function() {
      const serverAddress = document.getElementById("server-address").value;
      connect(serverAddress);
    };
    if (ilm.config.SERVER_ADDRESS !== null) {
      connect(ilm.config.SERVER_ADDRESS);
    }
  }

  document.addEventListener("DOMContentLoaded", initUi, false);
})(window.ilm);
