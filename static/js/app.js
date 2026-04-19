(function () {
    "use strict";

    const API_BASE = window.location.origin;
    let currentMode = "sd21";
    let cnControlnetImageBase64 = null;
    let cnReferenceImageBase64 = null;
    /** @type {Record<string, { label_zh?: string, hint_zh?: string, type?: string }>} */
    let referenceModesMeta = {};

    const LS_AUTOSPEAK = "mm-autospeak";

    function showToast(message, type) {
        type = type || "info";
        let region = document.getElementById("toast-region");
        if (!region) {
            region = document.createElement("div");
            region.id = "toast-region";
            region.setAttribute("aria-live", "polite");
            document.body.appendChild(region);
        }
        const el = document.createElement("div");
        el.className = "toast " + type;
        el.textContent = message;
        region.appendChild(el);
        const ms = type === "error" ? 6000 : 4000;
        setTimeout(function () {
            el.remove();
        }, ms);
    }

    function formatHttpError(data, status) {
        if (!data) return "HTTP " + status;
        const d = data.detail != null ? data.detail : data.error;
        if (d == null) return "HTTP " + status;
        if (typeof d === "string") return d;
        if (Array.isArray(d)) {
            return d
                .map(function (x) {
                    return x.msg || JSON.stringify(x);
                })
                .join("; ");
        }
        try {
            return JSON.stringify(d);
        } catch (_) {
            return String(d);
        }
    }

    window.switchMode = function (mode) {
        currentMode = mode;
        const tabSd = document.getElementById("tab-sd21");
        const tabCn = document.getElementById("tab-controlnet");
        const panelSd = document.getElementById("sd21-panel");
        const panelCn = document.getElementById("controlnet-panel");

        tabSd.classList.toggle("active", mode === "sd21");
        tabCn.classList.toggle("active", mode === "controlnet");
        tabSd.setAttribute("aria-selected", mode === "sd21" ? "true" : "false");
        tabCn.setAttribute("aria-selected", mode === "controlnet" ? "true" : "false");

        panelSd.classList.toggle("active", mode === "sd21");
        panelCn.classList.toggle("active", mode === "controlnet");

        tabSd.tabIndex = mode === "sd21" ? 0 : -1;
        tabCn.tabIndex = mode === "controlnet" ? 0 : -1;
    };

    function getPrefix() {
        return currentMode === "sd21" ? "sd21" : "cn";
    }

    function syncReferenceAdvancedControls() {
        const mode = document.getElementById("cn-ref-mode");
        const m = mode ? mode.value : "depth";
        const cnWrap = document.getElementById("cn-cn-scale-wrap");
        const ipWrap = document.getElementById("cn-ip-wrap");
        const stWrap = document.getElementById("cn-strength-wrap");
        const controlTypes = { depth: 1, openpose: 1, lineart: 1, softedge: 1, canny: 1 };
        if (cnWrap) cnWrap.style.display = controlTypes[m] ? "block" : "none";
        if (ipWrap) ipWrap.style.display = m === "ip_adapter" ? "block" : "none";
        if (stWrap) stWrap.style.display = m === "img2img" ? "block" : "none";
    }

    window.onReferenceModeChange = function () {
        const modeEl = document.getElementById("cn-ref-mode");
        const hintEl = document.getElementById("cn-ref-hint");
        const m = modeEl ? modeEl.value : "depth";
        const meta = referenceModesMeta[m];
        if (hintEl) {
            hintEl.textContent = meta && meta.hint_zh ? meta.hint_zh : "请选择模式并上传参考图。";
        }
        syncReferenceAdvancedControls();
        if (cnControlnetImageBase64) {
            cnControlnetImageBase64 = null;
            window.resetCannyUpload(true);
            showToast("已切换参考模式，请重新上传参考图", "info");
        }
    };

    window.resetCannyUpload = function (silent) {
        cnControlnetImageBase64 = null;
        cnReferenceImageBase64 = null;
        const prevRef = document.getElementById("cn-reference-preview");
        const prevCanny = document.getElementById("cn-canny-preview");
        const fileInput = document.getElementById("cn-reference-file");
        const drop = document.getElementById("cn-upload-dropzone");
        const status = document.getElementById("cn-upload-status");
        const resetBtn = document.getElementById("cn-reset-upload");
        if (prevRef) {
            prevRef.src = "";
            prevRef.classList.remove("visible");
        }
        if (prevCanny) {
            prevCanny.src = "";
            prevCanny.classList.remove("visible");
        }
        if (fileInput) fileInput.value = "";
        if (drop) drop.classList.remove("hidden");
        if (status) status.classList.add("hidden");
        if (resetBtn) resetBtn.classList.add("hidden");
        if (!silent) {
            showToast("已清除参考图，可重新上传", "info");
        }
    };

    window.handleReferenceUpload = function (event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const prefix = getPrefix();
        if (prefix !== "cn") return;

        const reader = new FileReader();
        reader.onload = async function (e) {
            cnReferenceImageBase64 = e.target.result.split(",")[1];
            const prevRef = document.getElementById("cn-reference-preview");
            const prevCanny = document.getElementById("cn-canny-preview");
            if (prevRef) {
                prevRef.src = e.target.result;
                prevRef.classList.add("visible");
            }

            const formData = new FormData();
            formData.append("file", file);
            const modeSel = document.getElementById("cn-ref-mode");
            formData.append("ref_mode", modeSel ? modeSel.value : "depth");

            try {
                const res = await fetch(API_BASE + "/process-reference", {
                    method: "POST",
                    body: formData,
                });
                let data;
                try {
                    data = await res.json();
                } catch (_) {
                    const text = await res.text();
                    throw new Error(text ? text.slice(0, 200) : "无效响应");
                }
                if (!res.ok) {
                    throw new Error(formatHttpError(data, res.status));
                }

                if (prevCanny && data.control_image) {
                    prevCanny.src = data.control_image;
                    prevCanny.classList.add("visible");
                }
                cnControlnetImageBase64 = data.control_image.split(",")[1];

                const drop = document.getElementById("cn-upload-dropzone");
                const status = document.getElementById("cn-upload-status");
                const resetBtn = document.getElementById("cn-reset-upload");
                if (drop) drop.classList.add("hidden");
                if (status) status.classList.remove("hidden");
                if (resetBtn) resetBtn.classList.remove("hidden");

                showToast("参考图已按当前模式处理，可点击「生成图像」", "success");
            } catch (err) {
                showToast("处理图片失败: " + err.message, "error");
                window.resetCannyUpload(true);
            }
        };
        reader.readAsDataURL(file);
    };

    window.enhancePrompt = async function (mode) {
        const prefix = mode;
        const promptEl = document.getElementById(prefix + "-prompt");
        const prompt = promptEl && promptEl.value.trim();
        if (!prompt) {
            showToast("请先输入提示词", "error");
            return;
        }

        const btn = document.getElementById(prefix + "-enhance-btn");
        const loading = document.getElementById(prefix + "-enhance-loading");
        const section = document.getElementById(prefix + "-enhanced-section");

        btn.disabled = true;
        if (loading) loading.style.display = "block";
        if (section) section.style.display = "none";

        try {
            const res = await fetch(API_BASE + "/enhance", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: prompt }),
            });
            let data;
            try {
                data = await res.json();
            } catch (_) {
                const text = await res.text();
                throw new Error(text ? text.slice(0, 200) : "empty response");
            }
            if (!res.ok) {
                throw new Error(formatHttpError(data, res.status));
            }
            if (!data.enhanced_prompt) {
                throw new Error("后端未返回扩充提示词");
            }
            const ta = document.getElementById(prefix + "-enhanced-prompt");
            if (ta) ta.value = data.enhanced_prompt;
            if (section) section.style.display = "block";
            showToast("提示词已扩充；生成时将优先使用下方扩充内容", "success");
        } catch (err) {
            showToast("扩充失败: " + err.message, "error");
        } finally {
            btn.disabled = false;
            if (loading) loading.style.display = "none";
        }
    };

    function setIndeterminateLoading(prefix, active) {
        const loading = document.getElementById(prefix + "-generate-loading");
        const bar = document.getElementById(prefix + "-progress-bar");
        if (!loading) return;
        if (active) {
            loading.classList.add("active");
            if (bar) bar.classList.add("indeterminate");
        } else {
            loading.classList.remove("active");
            if (bar) bar.classList.remove("indeterminate");
        }
    }

    window.generateImage = async function (mode) {
        const prefix = mode;
        const promptEl = document.getElementById(prefix + "-prompt");
        const prompt = promptEl && promptEl.value.trim();
        if (!prompt) {
            showToast("请先输入提示词", "error");
            return;
        }

        const btn = document.getElementById(prefix + "-generate-btn");
        const result = document.getElementById(prefix + "-generate-result");
        const placeholder = document.getElementById(prefix + "-placeholder-result");

        btn.disabled = true;
        setIndeterminateLoading(prefix, true);
        if (result) result.style.display = "none";
        if (placeholder) placeholder.style.display = "none";

        try {
            const enhancedPromptEl = document.getElementById(prefix + "-enhanced-prompt");
            const enhancedPrompt = enhancedPromptEl ? enhancedPromptEl.value.trim() : "";

            let requestBody = {
                prompt: prompt,
                num_inference_steps: parseInt(document.getElementById(prefix + "-steps").value, 10),
                guidance_scale: parseFloat(document.getElementById(prefix + "-guidance").value),
            };
            if (enhancedPrompt) {
                requestBody.enhanced_prompt = enhancedPrompt;
            }

            if (mode === "sd21") {
                const backend = document.getElementById("sd21-backend").value;
                if (backend === "sd21") {
                    requestBody.generation_mode = "sd21";
                } else {
                    requestBody.generation_mode = "sd15";
                    requestBody.sd15_style = backend;
                    const resSel = document.getElementById("sd15-resolution");
                    if (resSel) {
                        requestBody.sd15_resolution = parseInt(resSel.value, 10);
                    }
                }
            }

            let res;
            if (mode === "cn") {
                if (!cnControlnetImageBase64) {
                    showToast("请先在本页上传参考图（并等待处理完成）", "error");
                    btn.disabled = false;
                    setIndeterminateLoading(prefix, false);
                    if (placeholder) placeholder.style.display = "flex";
                    return;
                }
                const refMode = document.getElementById("cn-ref-mode")
                    ? document.getElementById("cn-ref-mode").value
                    : "depth";
                const refBody = {
                    ref_mode: refMode,
                    prompt: prompt,
                    control_image: cnControlnetImageBase64,
                    num_inference_steps: parseInt(document.getElementById(prefix + "-steps").value, 10),
                    guidance_scale: parseFloat(document.getElementById(prefix + "-guidance").value),
                    controlnet_conditioning_scale: parseFloat(
                        document.getElementById("cn-cn-scale").value
                    ),
                    ip_adapter_scale: parseFloat(document.getElementById("cn-ip-scale").value),
                    strength: parseFloat(document.getElementById("cn-strength").value),
                };
                if (enhancedPrompt) {
                    refBody.enhanced_prompt = enhancedPrompt;
                }
                res = await fetch(API_BASE + "/generate-reference", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(refBody),
                });
            } else {
                res = await fetch(API_BASE + "/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody),
                });
            }

            let data;
            try {
                data = await res.json();
            } catch (_) {
                const text = await res.text();
                throw new Error(text ? text.slice(0, 200) : "empty response");
            }
            if (!res.ok) {
                throw new Error(formatHttpError(data, res.status));
            }
            if (!data.image) {
                throw new Error("后端未返回图像数据");
            }

            document.getElementById(prefix + "-result-image").src = data.image;
            document.getElementById(prefix + "-result-caption").textContent = data.caption || "";

            if (data.used_num_steps && data.used_guidance) {
                document.getElementById(prefix + "-steps").value = data.used_num_steps;
                document.getElementById(prefix + "-steps-value").textContent = data.used_num_steps;
                document.getElementById(prefix + "-guidance").value = data.used_guidance;
                document.getElementById(prefix + "-guidance-value").textContent = data.used_guidance;
            }

            const clipEl = document.getElementById(prefix + "-clip-evaluation");
            if (data.clip_evaluation && clipEl) {
                const clip = data.clip_evaluation;
                document.getElementById(prefix + "-clip-quality").textContent =
                    "CLIP 语义相似度: " + (clip.similarity_score * 100).toFixed(1) + "%";
                document.getElementById(prefix + "-clip-similarity").textContent = clip.interpretation || "";
                clipEl.style.display = "block";
            } else if (clipEl) {
                clipEl.style.display = "none";
            }

            if (result) result.style.display = "block";
            if (placeholder) placeholder.style.display = "none";

            const autospeak = document.getElementById("prefs-autospeak");
            if (autospeak && autospeak.checked) {
                speakCaption(prefix);
            }
        } catch (err) {
            showToast("生成失败: " + err.message, "error");
            if (placeholder) placeholder.style.display = "flex";
        } finally {
            btn.disabled = false;
            setIndeterminateLoading(prefix, false);
        }
    };

    window.speakCaption = function (mode) {
        const prefix = mode;
        const captionEl = document.getElementById(prefix + "-result-caption");
        const caption = captionEl ? captionEl.textContent : "";
        if (caption) {
            const utterance = new SpeechSynthesisUtterance(caption);
            utterance.lang = "zh-CN";
            utterance.rate = 1.0;
            speechSynthesis.speak(utterance);
        }
    };

    window.downloadResultImage = function (prefix) {
        const img = document.getElementById(prefix + "-result-image");
        if (!img || !img.src) {
            showToast("没有可下载的图片", "error");
            return;
        }
        const a = document.createElement("a");
        a.href = img.src;
        a.download = "multimodal-generated.png";
        a.click();
        showToast("已开始下载", "success");
    };

    window.copyResultCaption = function (prefix) {
        const p = document.getElementById(prefix + "-result-caption");
        const text = p ? p.textContent : "";
        if (!text) {
            showToast("暂无描述可复制", "error");
            return;
        }
        navigator.clipboard.writeText(text).then(
            function () {
                showToast("描述已复制到剪贴板", "success");
            },
            function () {
                showToast("复制失败，请手动选择文字", "error");
            }
        );
    };

    function bindPrefs() {
        const cb = document.getElementById("prefs-autospeak");
        if (!cb) return;
        try {
            const saved = localStorage.getItem(LS_AUTOSPEAK);
            if (saved !== null) {
                cb.checked = saved === "true";
            }
        } catch (_) {}
        cb.addEventListener("change", function () {
            try {
                localStorage.setItem(LS_AUTOSPEAK, cb.checked ? "true" : "false");
            } catch (_) {}
        });
    }

    function bindKeyboard() {
        document.addEventListener("keydown", function (e) {
            if (!(e.ctrlKey || e.metaKey) || e.key !== "Enter") return;
            const t = e.target;
            if (!t || !t.id) return;
            if (t.id === "sd21-prompt" || t.id === "cn-prompt") {
                e.preventDefault();
                const mode = t.id === "sd21-prompt" ? "sd21" : "cn";
                window.generateImage(mode);
            }
        });
    }

    function syncSd15ResolutionVisibility() {
        const b = document.getElementById("sd21-backend");
        const w = document.getElementById("sd15-resolution-wrap");
        if (!b || !w) {
            return;
        }
        w.style.display = b.value === "sd21" ? "none" : "block";
    }

    document.addEventListener("DOMContentLoaded", function () {
        bindPrefs();
        bindKeyboard();
        document.getElementById("tab-sd21").setAttribute("aria-selected", "true");
        document.getElementById("tab-controlnet").setAttribute("aria-selected", "false");
        const backend = document.getElementById("sd21-backend");
        if (backend) {
            backend.addEventListener("change", syncSd15ResolutionVisibility);
            syncSd15ResolutionVisibility();
        }
        fetch(API_BASE + "/reference-modes")
            .then(function (r) {
                return r.json();
            })
            .then(function (cfg) {
                referenceModesMeta = cfg.modes || {};
                const sel = document.getElementById("cn-ref-mode");
                if (sel && cfg.modes) {
                    sel.innerHTML = "";
                    Object.keys(cfg.modes).forEach(function (key) {
                        const m = cfg.modes[key];
                        const opt = document.createElement("option");
                        opt.value = key;
                        opt.textContent = m.label_zh || key;
                        sel.appendChild(opt);
                    });
                }
                window.onReferenceModeChange();
            })
            .catch(function () {
                const hintEl = document.getElementById("cn-ref-hint");
                if (hintEl) {
                    hintEl.textContent =
                        "无法加载模式说明；仍可使用上方默认选项。Depth：场景深度；OpenPose：人物姿势；Lineart：线稿；SoftEdge：柔和边缘；Canny：硬边缘轮廓；IP-Adapter：风格参考；Img2Img：在参考图上重绘。";
                }
                syncReferenceAdvancedControls();
            });
    });
})();
