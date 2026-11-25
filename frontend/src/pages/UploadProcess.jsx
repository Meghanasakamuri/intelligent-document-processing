import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { uploadDocument } from "../services/api";

const MAX_UPLOAD_MB = 20;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"];
const ALLOWED_MIME_TYPES = [
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/jpg",
];

export default function UploadProcess() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  // File and Dropzone States
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState("");

  // Upload and Processing States
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStage, setUploadStage] = useState("idle"); // idle | uploading | completed | failed
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState("");

  // Validate selected file against actual backend constraints
  const validateFile = (selectedFile) => {
    if (!selectedFile) {
      return "No file selected.";
    }
    if (selectedFile.size === 0) {
      return "The selected file is empty (0 bytes). Please select a valid document.";
    }
    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      const sizeMB = (selectedFile.size / (1024 * 1024)).toFixed(1);
      return `File size (${sizeMB} MB) exceeds the maximum allowed limit of ${MAX_UPLOAD_MB} MB.`;
    }
    const ext = "." + (selectedFile.name.split(".").pop() || "").toLowerCase();
    const isExtensionAllowed = ALLOWED_EXTENSIONS.includes(ext);
    const isMimeAllowed = !selectedFile.type || ALLOWED_MIME_TYPES.includes(selectedFile.type);

    if (!isExtensionAllowed || !isMimeAllowed) {
      return `Unsupported file format "${ext}". Allowed formats: PDF, PNG, JPG, JPEG.`;
    }
    return null;
  };

  const handleSelectFile = (selectedFile) => {
    setValidationError("");
    setApiError("");
    setResult(null);
    setUploadStage("idle");

    if (!selectedFile) {
      setFile(null);
      return;
    }

    const errorMsg = validateFile(selectedFile);
    if (errorMsg) {
      setValidationError(errorMsg);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } else {
      setFile(selectedFile);
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0] || null;
    handleSelectFile(selected);
  };

  // Drag and Drop Handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      handleSelectFile(droppedFile);
    }
  };

  // Remove selected file
  const handleRemoveFile = () => {
    setFile(null);
    setValidationError("");
    setApiError("");
    setResult(null);
    setUploadStage("idle");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Upload and Process Document
  const handleUpload = async () => {
    if (!file || isUploading) return;

    const errorMsg = validateFile(file);
    if (errorMsg) {
      setValidationError(errorMsg);
      return;
    }

    setIsUploading(true);
    setUploadStage("uploading");
    setApiError("");
    setResult(null);

    try {
      const data = await uploadDocument(file);
      setResult(data);
      setUploadStage("completed");
    } catch (err) {
      setUploadStage("failed");
      setApiError(
        err.message ||
          "Unable to process document. Please make sure the backend server is running at http://127.0.0.1:8000."
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (!bytes && bytes !== 0) return "-";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const isPDF = file?.name?.toLowerCase().endsWith(".pdf");

  return (
    <div style={{ maxWidth: "1360px", margin: "0 auto", paddingBottom: "48px" }}>
      {/* Page Header */}
      <div style={{ marginBottom: "28px" }}>
        <h1
          style={{
            fontSize: "26px",
            fontWeight: "700",
            color: "var(--text-primary, #0f172a)",
            margin: "0 0 6px",
            letterSpacing: "-0.02em",
          }}
        >
          Upload & Ingest Document
        </h1>
        <p style={{ color: "var(--text-secondary, #475569)", fontSize: "14px", margin: 0, maxWidth: "720px", lineHeight: 1.5 }}>
          Ingest financial invoices, commercial receipts, and forms. Our pipeline performs automated OCR, document classification, heuristic extraction, and prepares tokens for LayoutLMv3 spatial modeling.
        </p>
      </div>

      {/* Main Grid: Upload Area & Pipeline Guide */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
          gap: "24px",
          marginBottom: "28px",
        }}
      >
        {/* Left Column: Drag & Drop Area */}
        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            padding: "26px",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          }}
        >
          <h2 style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a", margin: "0 0 16px" }}>
            Select Document
          </h2>

          {/* Hidden HTML File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            style={{ display: "none" }}
            disabled={isUploading}
          />

          {/* Large Dropzone */}
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => {
              if (!isUploading) fileInputRef.current?.click();
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                fileInputRef.current?.click();
              }
            }}
            style={{
              border: isDragging
                ? "2px dashed #4f46e5"
                : file
                ? "2px solid #cbd5e1"
                : "2px dashed #cbd5e1",
              borderRadius: "10px",
              backgroundColor: isDragging ? "#eef2ff" : file ? "#f8fafc" : "#fafafa",
              padding: "36px 20px",
              textAlign: "center",
              cursor: isUploading ? "not-allowed" : "pointer",
              transition: "all 0.2s ease",
              outline: "none",
            }}
          >
            <div
              style={{
                width: "56px",
                height: "56px",
                borderRadius: "50%",
                backgroundColor: isDragging ? "#e0e7ff" : "#f1f5f9",
                color: isDragging ? "#4f46e5" : "#64748b",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 16px",
                transition: "all 0.2s ease",
              }}
            >
              <UploadCloudIcon size={28} />
            </div>

            <div style={{ fontSize: "15px", fontWeight: "600", color: "#0f172a", marginBottom: "6px" }}>
              {isDragging ? "Drop your document here" : "Drag & drop document here, or click to browse"}
            </div>

            <p style={{ fontSize: "12px", color: "#64748b", margin: "0 auto 16px", maxWidth: "360px", lineHeight: 1.5 }}>
              Choose a file from your computer. Our backend parses text and spatial layout in seconds.
            </p>

            <button
              type="button"
              disabled={isUploading}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              style={{
                padding: "8px 18px",
                backgroundColor: "#ffffff",
                color: "var(--primary, #4f46e5)",
                border: "1px solid #c7d2fe",
                borderRadius: "6px",
                fontWeight: "600",
                fontSize: "13px",
                cursor: isUploading ? "not-allowed" : "pointer",
                boxShadow: "0 1px 2px rgba(15, 23, 42, 0.04)",
                transition: "all 0.15s ease",
              }}
            >
              Browse Files
            </button>
          </div>

          {/* Supported Format Information Pills */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "8px",
              marginTop: "14px",
              padding: "10px 14px",
              backgroundColor: "#f8fafc",
              borderRadius: "8px",
              border: "1px solid #e2e8f0",
              fontSize: "12px",
              color: "#64748b",
            }}
          >
            <div>
              <strong style={{ color: "#334155" }}>Formats: </strong>
              <span>PDF (up to 10 pages), PNG, JPG, JPEG</span>
            </div>
            <div>
              <strong style={{ color: "#334155" }}>Max size: </strong>
              <span>{MAX_UPLOAD_MB} MB</span>
            </div>
          </div>

          {/* Validation Error Alert */}
          {validationError && (
            <div
              style={{
                marginTop: "16px",
                padding: "12px 14px",
                backgroundColor: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: "8px",
                color: "#b91c1c",
                fontSize: "13px",
                display: "flex",
                alignItems: "flex-start",
                gap: "8px",
              }}
            >
              <AlertCircleIcon size={16} />
              <span>{validationError}</span>
            </div>
          )}

          {/* Selected File Card */}
          {file && (
            <div
              style={{
                marginTop: "18px",
                padding: "14px 16px",
                backgroundColor: "#ffffff",
                borderRadius: "8px",
                border: "1px solid #cbd5e1",
                boxShadow: "0 1px 2px rgba(15, 23, 42, 0.04)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "12px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "12px", minWidth: 0 }}>
                <div
                  style={{
                    width: "38px",
                    height: "38px",
                    borderRadius: "8px",
                    backgroundColor: isPDF ? "#fee2e2" : "#e0e7ff",
                    color: isPDF ? "#dc2626" : "#4f46e5",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  {isPDF ? <PdfFileIcon size={20} /> : <ImgFileIcon size={20} />}
                </div>

                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: "600",
                      color: "#0f172a",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {file.name}
                  </div>
                  <div style={{ fontSize: "11px", color: "#64748b", marginTop: "2px" }}>
                    {formatFileSize(file.size)} • {file.type || "Document"}
                  </div>
                </div>
              </div>

              {!isUploading && (
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  title="Remove file"
                  style={{
                    border: "none",
                    backgroundColor: "transparent",
                    color: "#94a3b8",
                    cursor: "pointer",
                    padding: "6px",
                    borderRadius: "4px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "color 0.15s ease",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = "#dc2626")}
                  onMouseLeave={(e) => (e.currentTarget.style.color = "#94a3b8")}
                >
                  <TrashIcon size={16} />
                </button>
              )}
            </div>
          )}

          {/* Upload Button */}
          <div style={{ marginTop: "20px" }}>
            <button
              type="button"
              onClick={handleUpload}
              disabled={isUploading || !file}
              style={{
                width: "100%",
                padding: "12px 20px",
                backgroundColor: isUploading || !file ? "#94a3b8" : "var(--primary, #4f46e5)",
                color: "#ffffff",
                border: "none",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "600",
                cursor: isUploading || !file ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "8px",
                boxShadow:
                  isUploading || !file
                    ? "none"
                    : "0 1px 3px rgba(79, 70, 229, 0.3), 0 1px 2px rgba(79, 70, 229, 0.2)",
                transition: "all 0.15s ease",
              }}
            >
              {isUploading ? (
                <>
                  <SpinnerIcon size={16} />
                  Processing Ingestion (OCR + Extraction)...
                </>
              ) : (
                <>
                  <UploadCloudIcon size={16} />
                  Upload & Process Document
                </>
              )}
            </button>
          </div>

          {/* Backend API Error Banner */}
          {apiError && (
            <div
              style={{
                marginTop: "16px",
                padding: "14px 16px",
                backgroundColor: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: "8px",
                color: "#b91c1c",
                fontSize: "13px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: "600", marginBottom: "4px" }}>
                <AlertCircleIcon size={16} />
                Upload Failed
              </div>
              <p style={{ margin: 0, fontSize: "12px", lineHeight: 1.4 }}>{apiError}</p>
            </div>
          )}
        </div>

        {/* Right Column: Processing Workflow Stages */}
        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            padding: "26px",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
          }}
        >
          <div>
            <h2 style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a", margin: "0 0 4px" }}>
              Processing Pipeline Stages
            </h2>
            <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 20px" }}>
              Backend-supported ingestion lifecycle for uploaded documents
            </p>

            {/* Stages List */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              {/* Stage 1: Document Upload */}
              <WorkflowStage
                number="01"
                title="Document Upload & Validation"
                description={
                  uploadStage === "completed"
                    ? `Saved to uploads directory (${result?.filename || file?.name})`
                    : uploadStage === "uploading"
                    ? "Transmitting bytes & validating file header..."
                    : "Validates format (.pdf, .png, .jpg) & size limit (< 20MB)"
                }
                status={
                  uploadStage === "completed"
                    ? "confirmed"
                    : uploadStage === "uploading"
                    ? "active"
                    : uploadStage === "failed"
                    ? "failed"
                    : "pending"
                }
              />

              {/* Stage 2: Hybrid Text Extraction */}
              <WorkflowStage
                number="02"
                title="Hybrid Text Extraction"
                description={
                  uploadStage === "completed"
                    ? `Extracted via ${result?.text_source || "OCR"} (${result?.pages_processed ?? 1} page${(result?.pages_processed ?? 1) > 1 ? "s" : ""} processed)`
                    : uploadStage === "uploading"
                    ? "Executing PyMuPDF embedded extraction & Tesseract OCR..."
                    : "PyMuPDF for digital text; Tesseract 5.4 for scanned images"
                }
                status={
                  uploadStage === "completed"
                    ? "confirmed"
                    : uploadStage === "uploading"
                    ? "active"
                    : "pending"
                }
              />

              {/* Stage 3: Classification & Baseline Extraction */}
              <WorkflowStage
                number="03"
                title="Document Classification & Extraction"
                description={
                  uploadStage === "completed"
                    ? `Classified as ${result?.document_type || "Invoice"} (Score: ${result?.coverage_score ?? result?.confidence_score ?? 0}%)`
                    : uploadStage === "uploading"
                    ? "Detecting document type & extracting canonical key fields..."
                    : "Keyword scoring (Invoice, Receipt, Form) + regex mapping"
                }
                status={
                  uploadStage === "completed"
                    ? "confirmed"
                    : uploadStage === "uploading"
                    ? "active"
                    : "pending"
                }
              />

              {/* Stage 4: LayoutLMv3 Advanced Extraction Ready */}
              <WorkflowStage
                number="04"
                title="Review & Spatial LayoutLMv3 Ready"
                description={
                  uploadStage === "completed"
                    ? "Saved to SQLite. Ready for human verification and LayoutLMv3"
                    : "Enables bounding box token classification & human verification"
                }
                status={uploadStage === "completed" ? "confirmed" : "pending"}
              />
            </div>
          </div>

          {/* Quick Notice */}
          <div
            style={{
              marginTop: "24px",
              padding: "12px 14px",
              backgroundColor: "#f8fafc",
              borderRadius: "8px",
              border: "1px solid #e2e8f0",
              fontSize: "12px",
              color: "#64748b",
              lineHeight: 1.4,
            }}
          >
            <strong style={{ color: "#334155" }}>Non-Simulated Processing: </strong>
            Stages advance only when verified and confirmed by the FastAPI backend responses.
          </div>
        </div>
      </div>

      {/* Success Result Confirmation Card */}
      {uploadStage === "completed" && result && (
        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "12px",
            border: "1px solid #a7f3d0",
            padding: "28px",
            boxShadow: "0 4px 6px -1px rgba(16, 185, 129, 0.08), 0 2px 4px -2px rgba(16, 185, 129, 0.05)",
          }}
        >
          {/* Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "flex-start",
              flexWrap: "wrap",
              gap: "12px",
              marginBottom: "20px",
              paddingBottom: "16px",
              borderBottom: "1px solid #e2e8f0",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <div
                style={{
                  width: "42px",
                  height: "42px",
                  borderRadius: "50%",
                  backgroundColor: "#ecfdf5",
                  color: "#15803d",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                <CheckCircleIcon size={24} />
              </div>
              <div>
                <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#0f172a", margin: "0 0 2px" }}>
                  Document Ingested Successfully
                </h2>
                <p style={{ fontSize: "13px", color: "#475569", margin: 0 }}>
                  File <strong style={{ color: "#0f172a" }}>{result.filename}</strong> has been saved and analyzed.
                </p>
              </div>
            </div>

            <span
              style={{
                padding: "4px 12px",
                borderRadius: "16px",
                backgroundColor: "#ecfdf5",
                color: "#15803d",
                border: "1px solid #a7f3d0",
                fontSize: "12px",
                fontWeight: "600",
              }}
            >
              Status: Processed
            </span>
          </div>

          {/* Document Ingestion Metadata Chips */}
          <div
            style={{
              display: "flex",
              gap: "10px",
              flexWrap: "wrap",
              marginBottom: "24px",
              fontSize: "12px",
            }}
          >
            <div style={{ padding: "6px 12px", backgroundColor: "#f8fafc", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Document Type: </span>
              <strong style={{ color: "#4f46e5" }}>{result.document_type || "Invoice"}</strong>
            </div>

            <div style={{ padding: "6px 12px", backgroundColor: "#f8fafc", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Text Engine: </span>
              <strong style={{ color: "#0f172a" }}>{result.text_source || "Tesseract OCR"}</strong>
            </div>

            <div style={{ padding: "6px 12px", backgroundColor: "#f8fafc", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Pages: </span>
              <strong style={{ color: "#0f172a" }}>{result.pages_processed ?? 1} / {result.page_count ?? 1}</strong>
            </div>

            <div style={{ padding: "6px 12px", backgroundColor: "#f8fafc", borderRadius: "6px", border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Initial Coverage: </span>
              <strong style={{ color: "#16a34a" }}>
                {result.coverage_score ?? result.confidence_score ?? 0}%
              </strong>
            </div>
          </div>

          {/* Extracted Fields Summary Grid */}
          <div style={{ marginBottom: "24px" }}>
            <h3 style={{ fontSize: "14px", fontWeight: "600", color: "#334155", margin: "0 0 12px" }}>
              Baseline Extracted Fields Preview
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
                gap: "12px",
              }}
            >
              <ExtractedFieldCard
                label="Vendor / Entity"
                value={result.invoice_fields?.vendor_name || result.invoice_fields?.company_name}
              />
              <ExtractedFieldCard
                label="Invoice / Doc #"
                value={result.invoice_fields?.invoice_number}
              />
              <ExtractedFieldCard
                label="Invoice Date"
                value={result.invoice_fields?.invoice_date}
              />
              <ExtractedFieldCard
                label="GSTIN / Tax ID"
                value={result.invoice_fields?.gst_number}
              />
              <ExtractedFieldCard
                label="Total Amount"
                value={
                  result.invoice_fields?.total_amount
                    ? `${result.invoice_fields?.currency || "₹"} ${result.invoice_fields?.total_amount}`
                    : result.invoice_fields?.subtotal
                    ? `${result.invoice_fields?.currency || "₹"} ${result.invoice_fields?.subtotal}`
                    : null
                }
              />
            </div>
          </div>

          {/* Navigation and Next Actions */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "12px",
              borderTop: "1px solid #e2e8f0",
              paddingTop: "20px",
            }}
          >
            <button
              type="button"
              onClick={handleRemoveFile}
              style={{
                padding: "10px 18px",
                backgroundColor: "#f8fafc",
                color: "#334155",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
                fontWeight: "600",
                fontSize: "13px",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <UploadCloudIcon size={15} />
              Upload Another Document
            </button>

            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
              <button
                type="button"
                onClick={() => navigate(`/qa`)}
                style={{
                  padding: "10px 18px",
                  backgroundColor: "#ffffff",
                  color: "#6d28d9",
                  border: "1px solid #ddd6fe",
                  borderRadius: "8px",
                  fontWeight: "600",
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                Ask Questions (QA)
              </button>

              <button
                type="button"
                onClick={() => navigate(`/review/${result.id}`)}
                style={{
                  padding: "10px 22px",
                  backgroundColor: "var(--primary, #4f46e5)",
                  color: "#ffffff",
                  border: "none",
                  borderRadius: "8px",
                  fontWeight: "600",
                  fontSize: "13px",
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  boxShadow: "0 1px 3px rgba(79, 70, 229, 0.25)",
                }}
              >
                Open Document Review (LayoutLMv3 & Verify)
                <span style={{ fontSize: "14px" }}>&rarr;</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Reusable Component: Workflow Stage Row ───────────────────────────
function WorkflowStage({ number, title, description, status }) {
  const isConfirmed = status === "confirmed";
  const isActive = status === "active";
  const isFailed = status === "failed";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "12px",
        padding: "12px",
        borderRadius: "8px",
        backgroundColor: isConfirmed ? "#f0fdf4" : isActive ? "#eef2ff" : isFailed ? "#fef2f2" : "#f8fafc",
        border: `1px solid ${
          isConfirmed ? "#bbf7d0" : isActive ? "#c7d2fe" : isFailed ? "#fecaca" : "#e2e8f0"
        }`,
      }}
    >
      <div
        style={{
          width: "28px",
          height: "28px",
          borderRadius: "50%",
          backgroundColor: isConfirmed ? "#16a34a" : isActive ? "#4f46e5" : isFailed ? "#dc2626" : "#e2e8f0",
          color: isConfirmed || isActive || isFailed ? "#ffffff" : "#64748b",
          fontSize: "11px",
          fontWeight: "700",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {isConfirmed ? (
          <CheckMarkIcon size={14} />
        ) : isActive ? (
          <SpinnerIcon size={14} />
        ) : (
          number
        )}
      </div>

      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2px" }}>
          <h4 style={{ fontSize: "13px", fontWeight: "600", color: "#0f172a", margin: 0 }}>
            {title}
          </h4>
          <span
            style={{
              fontSize: "10px",
              fontWeight: "600",
              padding: "2px 6px",
              borderRadius: "4px",
              textTransform: "uppercase",
              backgroundColor: isConfirmed
                ? "#dcfce7"
                : isActive
                ? "#e0e7ff"
                : isFailed
                ? "#fee2e2"
                : "#f1f5f9",
              color: isConfirmed
                ? "#15803d"
                : isActive
                ? "#4338ca"
                : isFailed
                ? "#b91c1c"
                : "#64748b",
            }}
          >
            {isConfirmed ? "Confirmed" : isActive ? "Processing" : isFailed ? "Failed" : "Pending"}
          </span>
        </div>
        <p style={{ fontSize: "12px", color: "#64748b", margin: 0, lineHeight: 1.4 }}>
          {description}
        </p>
      </div>
    </div>
  );
}

// ── Reusable Component: Extracted Field Card ─────────────────────────
function ExtractedFieldCard({ label, value }) {
  const displayVal = value && value !== "Not found" ? value : "Not detected";
  const isDetected = displayVal !== "Not detected";

  return (
    <div
      style={{
        padding: "12px 14px",
        backgroundColor: "#f8fafc",
        borderRadius: "8px",
        border: "1px solid #e2e8f0",
      }}
    >
      <div style={{ fontSize: "11px", fontWeight: "500", color: "#64748b", marginBottom: "4px" }}>
        {label}
      </div>
      <div
        style={{
          fontSize: "13px",
          fontWeight: isDetected ? "600" : "400",
          color: isDetected ? "#0f172a" : "#94a3b8",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
        title={displayVal}
      >
        {displayVal}
      </div>
    </div>
  );
}

// ── Vector SVG Icons ─────────────────────────────────────────────────
function UploadCloudIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
      <path d="M12 12v9" />
      <path d="m16 16-4-4-4 4" />
    </svg>
  );
}

function PdfFileIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M9 15h2a1.5 1.5 0 0 0 0-3H9v6" />
    </svg>
  );
}

function ImgFileIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

function CheckCircleIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function CheckMarkIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AlertCircleIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

function TrashIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function SpinnerIcon({ size = 16 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        animation: "spin 1s linear infinite",
      }}
    >
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </svg>
  );
}
