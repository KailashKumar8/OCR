const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const previewWrap = document.getElementById("previewWrap");
    const previewImage = document.getElementById("previewImage");
    const previewLink = document.getElementById("previewLink");
    const previewOpen = document.getElementById("previewOpen");
    const uploadArea = document.getElementById("uploadArea");
    const ocrForm = document.getElementById("ocrForm");
    const fileError = document.getElementById("fileError");
    const resultText = document.getElementById("resultText");
    const copyBtn = document.getElementById("copyBtn");
    const editToggleBtn = document.getElementById("editToggleBtn");
    const resetTextBtn = document.getElementById("resetTextBtn");
    const editStateLabel = document.getElementById("editStateLabel");
    const editedTextCanvas = document.getElementById("editedTextCanvas");
    const editedPreviewOpen = document.getElementById("editedPreviewOpen");
    const downloadEditedImage = document.getElementById("downloadEditedImage");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const placeholderText = "Extracted text will appear here after you run OCR.";
    let originalExtractedText = "";

    function normalizedResultText() {
      const raw = resultText.innerText.replace(/\u00A0/g, " ").trim();
      return raw === placeholderText ? "" : raw;
    }

    function setEditMode(enabled) {
      resultText.contentEditable = enabled ? "true" : "false";
      resultText.classList.toggle("editing", enabled);
      editToggleBtn.innerText = enabled ? "Finish Edit" : "Manual Edit";
      editToggleBtn.classList.toggle("active", enabled);
      resetTextBtn.classList.toggle("is-hidden", !enabled);
      editStateLabel.innerText = enabled ? "Editing Enabled" : "Read Only";

      if (enabled && normalizedResultText() === "") {
        resultText.innerText = "";
      }
      if (enabled) {
        resultText.focus();
      }
      updateStats();
    }

    function wrapForCanvas(ctx, paragraph, maxWidth) {
      const words = paragraph.split(/\s+/).filter(Boolean);
      if (!words.length) {
        return [""];
      }
      const lines = [];
      let line = words[0];
      for (let i = 1; i < words.length; i++) {
        const next = `${line} ${words[i]}`;
        if (ctx.measureText(next).width <= maxWidth) {
          line = next;
        } else {
          lines.push(line);
          line = words[i];
        }
      }
      lines.push(line);
      return lines;
    }

    function renderEditedHandwritingPreview() {
      if (!editedTextCanvas) {
        return;
      }

      const text = normalizedResultText() || "No text available for manual edit preview.";
      const ctx = editedTextCanvas.getContext("2d");
      const width = 980;
      const sidePad = 76;
      const topPad = 68;
      const lineHeight = 44;
      const maxTextWidth = width - sidePad * 2;

      ctx.font = "700 40px 'Caveat', 'Comic Sans MS', cursive";
      const allLines = [];
      const paragraphs = text.split(/\n/);
      for (const paragraph of paragraphs) {
        if (!paragraph.trim()) {
          allLines.push("");
          continue;
        }
        allLines.push(...wrapForCanvas(ctx, paragraph, maxTextWidth));
      }

      const height = Math.max(620, topPad + allLines.length * lineHeight + 44);
      editedTextCanvas.width = width;
      editedTextCanvas.height = height;

      ctx.fillStyle = "#f8f3e4";
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = "rgba(98, 145, 201, 0.22)";
      ctx.lineWidth = 1;
      for (let y = topPad + 12; y < height; y += lineHeight) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.strokeStyle = "rgba(220, 93, 93, 0.36)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(sidePad - 20, 0);
      ctx.lineTo(sidePad - 20, height);
      ctx.stroke();

      ctx.fillStyle = "#1f3550";
      ctx.font = "700 40px 'Caveat', 'Comic Sans MS', cursive";
      let y = topPad;
      allLines.forEach((line, idx) => {
        if (!line) {
          y += lineHeight * 0.7;
          return;
        }
        const jitter = (idx % 4) * 0.6;
        const x = sidePad + ((idx % 2 === 0) ? jitter : -jitter);
        ctx.fillText(line, x, y);
        y += lineHeight;
      });

      const png = editedTextCanvas.toDataURL("image/png");
      if (editedPreviewOpen) {
        editedPreviewOpen.href = png;
      }
      if (downloadEditedImage) {
        downloadEditedImage.href = png;
      }
    }

    function updateStats() {
      const value = normalizedResultText();
      document.getElementById("charCount").innerText = value.length;
      document.getElementById("wordCount").innerText = value ? value.split(/\s+/).length : 0;
      document.getElementById("lineCount").innerText = value ? value.split(/\n+/).filter(Boolean).length : 0;
      copyBtn.disabled = !value;
      if (value) {
        resultText.classList.remove("result-empty");
      } else if (resultText.contentEditable !== "true") {
        resultText.classList.add("result-empty");
      }
      renderEditedHandwritingPreview();
    }

    function showPreview(file) {
      if (!file) {
        previewLink.href = "#";
        previewOpen.href = "#";
        previewWrap.classList.remove("show");
        return;
      }
      const url = URL.createObjectURL(file);
      previewImage.src = url;
      previewLink.href = url;
      previewOpen.href = url;
      previewWrap.classList.add("show");
    }

    function handleFile(file) {
      if (!file) {
        fileName.innerText = "No file selected";
        previewWrap.classList.remove("show");
        return;
      }
      const validTypes = ["image/png", "image/jpeg", "image/jpg"];
      if (!validTypes.includes(file.type)) {
        fileError.style.display = "block";
        fileName.innerText = "Unsupported file type";
        previewWrap.classList.remove("show");
        return;
      }
      fileError.style.display = "none";
      fileName.innerText = file.name;
      showPreview(file);
    }

    fileInput.addEventListener("change", () => {
      handleFile(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadArea.classList.add("drag-active");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadArea.classList.remove("drag-active");
      });
    });

    uploadArea.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (!file) {
        return;
      }
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      handleFile(file);
    });

    ocrForm.addEventListener("submit", (event) => {
      if (!fileInput.files.length) {
        event.preventDefault();
        fileError.style.display = "block";
        return;
      }
      loadingOverlay.classList.add("show");
    });

    editToggleBtn.addEventListener("click", () => {
      setEditMode(resultText.contentEditable !== "true");
    });

    resetTextBtn.addEventListener("click", () => {
      if (originalExtractedText) {
        resultText.innerText = originalExtractedText;
        resultText.classList.remove("result-empty");
      } else {
        resultText.innerText = placeholderText;
        resultText.classList.add("result-empty");
      }
      setEditMode(false);
      updateStats();
    });

    resultText.addEventListener("input", () => {
      updateStats();
    });

    copyBtn.addEventListener("click", async () => {
      const content = normalizedResultText();
      if (!content) {
        return;
      }
      await navigator.clipboard.writeText(content);
      const original = copyBtn.innerText;
      copyBtn.innerText = "Copied";
      setTimeout(() => {
        copyBtn.innerText = original;
      }, 1200);
    });

    originalExtractedText = normalizedResultText();
    setEditMode(false);
    updateStats();
