import { useEffect, useState, useMemo } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  getDocument,
  getDocuments,
  updateDocument,
  verifyDocument,
  saveGroundTruth,
  getDocumentFileUrl,
  runLayoutLMExtraction,
  getAuditLog,
} from "../services/api";

export default function DocumentReview() {
  const { id: paramId } = useParams();
  const navigate = useNavigate();

  // Document & History Data
  const [doc, setDoc] = useState(null);
  const [docList, setDocList] = useState([]);
  const [auditLog, setAuditLog] = useState([]);
  const [loading, setLoading] = useState(true);
  const [feedback, setFeedback] = useState({ type: "", message: "" });

  // Right Panel Tab: "edit" | "compare" | "audit"
  const [activeTab, setActiveTab] = useState("edit");

  // Form State
  const [fields, setFields] = useState({});
  const [verified, setVerified] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [saving, setSaving] = useState(false);
  const [runningModel, setRunningModel] = useState(false);

  // Left Panel Preview Controls
  const [zoom, setZoom] = useState(100);
  const [currentPage, setCurrentPage] = useState(1);

  // Warn user before leaving page with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [hasUnsavedChanges]);

  // Load document and document list
  useEffect(() => {
    let active = true;

    async function fetchDocument() {
      try {
        let activeId = paramId;

        // If no ID in route, fetch most recent document
        if (!activeId) {
          const recent = await getDocuments({ limit: 1 }).catch(() => ({ documents: [] }));
          if (recent.documents && recent.documents.length > 0) {
            activeId = recent.documents[0].id;
          } else {
            if (active) {
              setDoc(null);
              setLoading(false);
            }
            return;
          }
        }

        const [data, logs, allDocs] = await Promise.all([
          getDocument(activeId),
          getAuditLog(activeId).catch(() => []),
          getDocuments({ limit: 25 }).catch(() => ({ documents: [] })),
        ]);

        if (active) {
          setDoc(data);
          setVerified(Boolean(data.verified));
          setAuditLog(logs || []);
          setDocList(allDocs?.documents || []);
          setCurrentPage(1);

          // Determine active fields
          const isLayoutLM =
            data.extraction_method === "layoutlm" ||
            Boolean(data.layoutlm_fields && Object.keys(data.layoutlm_fields).length > 0);

          const sourceFields = isLayoutLM && data.layoutlm_fields
            ? data.layoutlm_fields
            : (data.invoice_fields || data.extracted_fields || {});

          // If already human-verified, preserve the verified invoice_fields; otherwise take the source
          const initialFields = data.verified && data.invoice_fields && Object.keys(data.invoice_fields).length > 0
            ? data.invoice_fields
            : sourceFields;

          setFields({
            vendor_name: initialFields.vendor_name || initialFields.company_name || "",
            customer_name: initialFields.customer_name || "",
            gst_number: initialFields.gst_number || "",
            customer_gst: initialFields.customer_gst || initialFields.buyer_gstin || initialFields.customer_gstin || "",
            invoice_number: initialFields.invoice_number || "",
            invoice_date: initialFields.invoice_date || "",
            due_date: initialFields.due_date || "",
            gross_amount: initialFields.gross_amount || "",
            subtotal: initialFields.subtotal || "",
            cgst_amount: initialFields.cgst_amount || "",
            sgst_amount: initialFields.sgst_amount || "",
            tax_amount: initialFields.tax_amount || "",
            total_amount: initialFields.total_amount || "",
            currency: initialFields.currency || "",
          });
          setHasUnsavedChanges(false);
        }
      } catch (err) {
        if (active) {
          setFeedback({ type: "error", message: err.message || "Failed to load document." });
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    fetchDocument();

    return () => {
      active = false;
    };
  }, [paramId]);

  // Update a field value
  const handleFieldChange = (key, value) => {
    setFields((prev) => ({
      ...prev,
      [key]: value,
      // Keep vendor_name and company_name in sync for backward compatibility
      ...(key === "vendor_name" ? { company_name: value } : {}),
    }));
    setHasUnsavedChanges(true);
  };

  // Save changes without changing verification status
  const handleSaveOnly = async () => {
    if (!doc || saving) return;
    setSaving(true);
    setFeedback({ type: "", message: "" });

    try {
      const updated = await updateDocument(doc.id, { invoice_fields: fields });
      const logs = await getAuditLog(doc.id).catch(() => []);
      setDoc(updated);
      setAuditLog(logs);
      setHasUnsavedChanges(false);
      setFeedback({ type: "success", message: "Field changes saved successfully and recorded in audit log." });
    } catch (err) {
      setFeedback({ type: "error", message: err.message || "Failed to save field modifications." });
    } finally {
      setSaving(false);
    }
  };

  // Save changes & Approve Document
  const handleApprove = async () => {
    if (!doc || saving) return;
    setSaving(true);
    setFeedback({ type: "", message: "" });

    try {
      await updateDocument(doc.id, { invoice_fields: fields });
      const updated = await verifyDocument(doc.id, true);
      const logs = await getAuditLog(doc.id).catch(() => []);

      setDoc(updated);
      setVerified(true);
      setAuditLog(logs);
      setHasUnsavedChanges(false);
      setFeedback({ type: "success", message: "Document verified & approved. Verified state recorded in database." });
    } catch (err) {
      setFeedback({ type: "error", message: err.message || "Approval failed." });
    } finally {
      setSaving(false);
    }
  };

  // Reject / Invalidate Verification
  const handleReject = async () => {
    if (!doc || saving) return;
    setSaving(true);
    setFeedback({ type: "", message: "" });

    try {
      const updated = await verifyDocument(doc.id, false);
      const logs = await getAuditLog(doc.id).catch(() => []);

      setDoc(updated);
      setVerified(false);
      setAuditLog(logs);
      setFeedback({ type: "info", message: "Document marked unverified. Requires further manual inspection." });
    } catch (err) {
      setFeedback({ type: "error", message: err.message || "Action failed." });
    } finally {
      setSaving(false);
    }
  };

  // Run LayoutLMv3 Extraction on demand
  const handleRunLayoutLM = async () => {
    if (!doc || runningModel) return;
    setRunningModel(true);
    setFeedback({ type: "", message: "" });

    try {
      const res = await runLayoutLMExtraction(doc.id);
      const updated = await getDocument(doc.id);
      setDoc(updated);

      const newFields = res.fields || updated.layoutlm_fields || {};
      setFields((prev) => ({
        ...prev,
        ...newFields,
        vendor_name: newFields.vendor_name || newFields.company_name || prev.vendor_name,
        customer_name: newFields.customer_name || prev.customer_name,
        gst_number: newFields.gst_number || prev.gst_number,
        customer_gst: newFields.customer_gst || newFields.buyer_gstin || prev.customer_gst,
        invoice_number: newFields.invoice_number || prev.invoice_number,
        invoice_date: newFields.invoice_date || prev.invoice_date,
        due_date: newFields.due_date || prev.due_date,
        gross_amount: newFields.gross_amount || prev.gross_amount,
        subtotal: newFields.subtotal || prev.subtotal,
        cgst_amount: newFields.cgst_amount || prev.cgst_amount,
        sgst_amount: newFields.sgst_amount || prev.sgst_amount,
        tax_amount: newFields.tax_amount || prev.tax_amount,
        total_amount: newFields.total_amount || prev.total_amount,
        currency: newFields.currency || prev.currency,
      }));

      setHasUnsavedChanges(true);
      setFeedback({
        type: "success",
        message: "LayoutLMv3 spatial token classification completed. Values loaded into review fields.",
      });
    } catch (err) {
      setFeedback({ type: "error", message: err.message || "LayoutLMv3 inference failed." });
    } finally {
      setRunningModel(false);
    }
  };

  // Save current values as ground truth for evaluation
  const handleSaveGroundTruth = async () => {
    if (!doc || saving) return;
    setSaving(true);
    setFeedback({ type: "", message: "" });

    try {
      await saveGroundTruth(doc.id, fields);
      setFeedback({
        type: "success",
        message: "Current fields saved as Ground Truth for benchmark accuracy/F1 evaluation.",
      });
    } catch (err) {
      setFeedback({ type: "error", message: err.message || "Failed to save ground truth." });
    } finally {
      setSaving(false);
    }
  };

  // Compute LayoutLMv3 confidence scores map
  const confidenceMap = useMemo(() => {
    if (!doc) return {};
    return doc.layoutlm_confidence || {};
  }, [doc]);

  // Baseline vs LayoutLM comparison data
  const comparisonList = useMemo(() => {
    if (!doc) return [];

    const baseline = doc.baseline_fields || doc.baseline_invoice_fields || doc.extracted_fields || {};
    const layoutlm = doc.layoutlm_fields || {};

    const fieldKeys = [
      { key: "vendor_name", label: "Vendor Name" },
      { key: "customer_name", label: "Customer Name" },
      { key: "invoice_number", label: "Invoice Number" },
      { key: "invoice_date", label: "Invoice Date" },
      { key: "due_date", label: "Due Date" },
      { key: "gst_number", label: "Vendor GSTIN" },
      { key: "customer_gst", label: "Customer GSTIN" },
      { key: "subtotal", label: "Subtotal Amount" },
      { key: "cgst_amount", label: "CGST Amount" },
      { key: "sgst_amount", label: "SGST Amount" },
      { key: "tax_amount", label: "Total Tax Amount" },
      { key: "total_amount", label: "Total / Net Amount" },
      { key: "currency", label: "Currency" },
    ];

    return fieldKeys.map(({ key, label }) => {
      const bVal = baseline[key] || (key === "vendor_name" ? baseline.company_name : null) || null;
      const lVal = layoutlm[key] || (key === "vendor_name" ? layoutlm.company_name : null) || null;

      const normB = bVal ? String(bVal).trim().toLowerCase() : "";
      const normL = lVal ? String(lVal).trim().toLowerCase() : "";

      let status = "match";
      if (!normB && !normL) {
        status = "both_empty";
      } else if (!normB && normL) {
        status = "layoutlm_only";
      } else if (normB && !normL) {
        status = "baseline_only";
      } else if (normB !== normL) {
        status = "mismatch";
      }

      return {
        key,
        label,
        baselineValue: bVal,
        layoutlmValue: lVal,
        status,
        confidence: confidenceMap[key],
      };
    });
  }, [doc, confidenceMap]);

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "80px 20px", color: "#64748b" }}>
        <div style={{ display: "inline-block", marginBottom: "12px" }}>
          <SpinnerIcon size={32} />
        </div>
        <h2 style={{ fontSize: "18px", color: "#0f172a", margin: 0 }}>Loading document review...</h2>
        <p style={{ fontSize: "13px", color: "#64748b", marginTop: "6px" }}>Fetching metadata, preview, and audit trail</p>
      </div>
    );
  }

  if (!doc) {
    return (
      <div style={{ maxWidth: "600px", margin: "60px auto", textAlign: "center" }}>
        <div
          style={{
            backgroundColor: "#ffffff",
            padding: "40px",
            borderRadius: "12px",
            border: "1px solid #e2e8f0",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          }}
        >
          <div style={{ width: "48px", height: "48px", borderRadius: "50%", backgroundColor: "#f1f5f9", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px", color: "#94a3b8" }}>
            <FileTextIcon size={24} />
          </div>
          <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#0f172a", margin: "0 0 6px" }}>No Document Selected</h2>
          <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 20px" }}>
            Upload a document or choose one from the recent documents list to begin review.
          </p>
          <Link
            to="/upload"
            style={{
              display: "inline-block",
              padding: "10px 20px",
              backgroundColor: "var(--primary, #4f46e5)",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "13px",
            }}
          >
            Upload Document &rarr;
          </Link>
        </div>
      </div>
    );
  }

  const isPdf = (doc.filename || "").toLowerCase().endsWith(".pdf");
  const fileUrl = getDocumentFileUrl(doc.id);
  const isLayoutLM =
    doc.extraction_method === "layoutlm" ||
    Boolean(doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0);
  const totalPages = doc.page_count ?? 1;
  const validationResults = doc.validation_results || null;
  const dateValidation = validationResults?.dates || {};
  const lineItems = Array.isArray(doc.line_items) ? doc.line_items : [];

  return (
    <div style={{ maxWidth: "1360px", margin: "0 auto", paddingBottom: "40px" }}>
      {/* 1. Header & Switcher Toolbar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "16px",
          marginBottom: "20px",
          paddingBottom: "16px",
          borderBottom: "1px solid var(--border-subtle, #e2e8f0)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#0f172a", margin: 0, letterSpacing: "-0.02em" }}>
              {doc.filename}
            </h1>

            {/* Verification Status Pill */}
            <span
              style={{
                fontSize: "11px",
                fontWeight: "600",
                padding: "3px 9px",
                borderRadius: "12px",
                backgroundColor: verified ? "#ecfdf5" : "#fff7ed",
                color: verified ? "#15803d" : "#c2410c",
                border: `1px solid ${verified ? "#a7f3d0" : "#fed7aa"}`,
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              {verified ? "✓ Verified" : "⚠ Pending Verification"}
            </span>

            {/* Unsaved Changes Indicator */}
            {hasUnsavedChanges && (
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "600",
                  padding: "3px 8px",
                  borderRadius: "12px",
                  backgroundColor: "#fef3c7",
                  color: "#b45309",
                  border: "1px solid #fde68a",
                }}
              >
                ● Unsaved Changes
              </span>
            )}
          </div>
        </div>

        {/* Action Controls & Document Switcher */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
          {/* Quick Document Switcher */}
          {docList.length > 1 && (
            <select
              value={doc.id}
              onChange={(e) => navigate(`/review/${e.target.value}`)}
              style={{
                padding: "8px 12px",
                fontSize: "12px",
                border: "1px solid #cbd5e1",
                borderRadius: "6px",
                backgroundColor: "#ffffff",
                color: "#334155",
                outline: "none",
                cursor: "pointer",
              }}
            >
              {docList.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} ({d.verified ? "Verified" : "Unverified"})
                </option>
              ))}
            </select>
          )}

          {/* Run LayoutLMv3 */}
          <button
            type="button"
            onClick={handleRunLayoutLM}
            disabled={runningModel}
            style={{
              padding: "8px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: runningModel ? "#e2e8f0" : "#f5f3ff",
              color: runningModel ? "#94a3b8" : "#6d28d9",
              border: "1px solid #ddd6fe",
              borderRadius: "6px",
              cursor: runningModel ? "not-allowed" : "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <SparklesIcon size={14} />
            {runningModel ? "Running LayoutLMv3..." : "⚡ Run LayoutLMv3"}
          </button>

          {/* Save Changes */}
          <button
            type="button"
            onClick={handleSaveOnly}
            disabled={saving || !hasUnsavedChanges}
            style={{
              padding: "8px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: hasUnsavedChanges ? "#ffffff" : "#f8fafc",
              color: hasUnsavedChanges ? "#0f172a" : "#94a3b8",
              border: "1px solid #cbd5e1",
              borderRadius: "6px",
              cursor: hasUnsavedChanges ? "pointer" : "default",
            }}
          >
            {saving ? "Saving..." : "Save Draft"}
          </button>

          {/* Reject / Invalidate (if verified) */}
          {verified && (
            <button
              type="button"
              onClick={handleReject}
              disabled={saving}
              style={{
                padding: "8px 14px",
                fontSize: "12px",
                fontWeight: "600",
                backgroundColor: "#fff7ed",
                color: "#c2410c",
                border: "1px solid #fed7aa",
                borderRadius: "6px",
                cursor: saving ? "not-allowed" : "pointer",
              }}
            >
              Reopen / Invalidate
            </button>
          )}

          {/* Approve & Verify Button */}
          <button
            type="button"
            onClick={handleApprove}
            disabled={saving}
            style={{
              padding: "8px 18px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: verified ? "#15803d" : "var(--primary, #4f46e5)",
              color: "#ffffff",
              border: "none",
              borderRadius: "6px",
              cursor: saving ? "not-allowed" : "pointer",
              boxShadow: "0 1px 2px rgba(15, 23, 42, 0.08)",
            }}
          >
            {saving ? "Saving..." : verified ? "✓ Re-Approve Changes" : "✓ Approve & Verify"}
          </button>
        </div>
      </div>

      {/* Feedback Banner */}
      {feedback.message && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "8px",
            marginBottom: "20px",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            backgroundColor:
              feedback.type === "success"
                ? "#ecfdf5"
                : feedback.type === "info"
                ? "#eff6ff"
                : "#fef2f2",
            border: `1px solid ${
              feedback.type === "success"
                ? "#a7f3d0"
                : feedback.type === "info"
                ? "#bfdbfe"
                : "#fecaca"
            }`,
            color:
              feedback.type === "success"
                ? "#166534"
                : feedback.type === "info"
                ? "#1e40af"
                : "#991b1b",
          }}
        >
          <span>{feedback.message}</span>
          <button
            onClick={() => setFeedback({ type: "", message: "" })}
            style={{ border: "none", background: "none", color: "inherit", cursor: "pointer", fontSize: "14px" }}
          >
            &times;
          </button>
        </div>
      )}

      {/* 2. Main Two-Panel Workspace Grid */}
      <div className="review-workspace-grid">
        {/* ── LEFT PANEL: Original Document Preview ── */}
        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            padding: "18px 20px",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
            display: "flex",
            flexDirection: "column",
            height: "calc(100vh - 200px)",
            minHeight: "680px",
          }}
        >
          {/* Preview Toolbar */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "12px",
              paddingBottom: "10px",
              borderBottom: "1px solid #f1f5f9",
              flexWrap: "wrap",
              gap: "8px",
            }}
          >
            {/* Document Info */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "13px", fontWeight: "600", color: "#0f172a" }}>
                Original Document
              </span>
              <span
                style={{
                  fontSize: "11px",
                  padding: "2px 7px",
                  borderRadius: "4px",
                  backgroundColor: "#f1f5f9",
                  color: "#475569",
                  fontWeight: "600",
                }}
              >
                {isPdf ? "PDF" : "IMAGE"}
              </span>
            </div>

            {/* Page & Zoom Controls */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              {/* Multi-page Navigation */}
              {isPdf && totalPages > 1 && (
                <div style={{ display: "flex", alignItems: "center", gap: "4px", marginRight: "6px" }}>
                  <button
                    type="button"
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage <= 1}
                    style={{
                      border: "1px solid #cbd5e1",
                      backgroundColor: "#ffffff",
                      borderRadius: "4px",
                      padding: "3px 6px",
                      cursor: currentPage <= 1 ? "not-allowed" : "pointer",
                      color: currentPage <= 1 ? "#cbd5e1" : "#334155",
                      fontSize: "11px",
                    }}
                  >
                    &larr;
                  </button>
                  <span style={{ fontSize: "11px", color: "#64748b", minWidth: "50px", textAlign: "center" }}>
                    {currentPage} / {totalPages}
                  </span>
                  <button
                    type="button"
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage >= totalPages}
                    style={{
                      border: "1px solid #cbd5e1",
                      backgroundColor: "#ffffff",
                      borderRadius: "4px",
                      padding: "3px 6px",
                      cursor: currentPage >= totalPages ? "not-allowed" : "pointer",
                      color: currentPage >= totalPages ? "#cbd5e1" : "#334155",
                      fontSize: "11px",
                    }}
                  >
                    &rarr;
                  </button>
                </div>
              )}

              {/* Zoom Controls */}
              <div style={{ display: "flex", alignItems: "center", gap: "2px", border: "1px solid #cbd5e1", borderRadius: "4px", padding: "1px" }}>
                <button
                  type="button"
                  title="Zoom Out"
                  onClick={() => setZoom((z) => Math.max(50, z - 15))}
                  style={{ border: "none", background: "none", cursor: "pointer", padding: "3px 7px", fontSize: "12px", color: "#475569" }}
                >
                  -
                </button>
                <span style={{ fontSize: "11px", color: "#334155", padding: "0 4px", minWidth: "36px", textAlign: "center" }}>
                  {zoom}%
                </span>
                <button
                  type="button"
                  title="Zoom In"
                  onClick={() => setZoom((z) => Math.min(200, z + 15))}
                  style={{ border: "none", background: "none", cursor: "pointer", padding: "3px 7px", fontSize: "12px", color: "#475569" }}
                >
                  +
                </button>
                <button
                  type="button"
                  title="Reset Zoom"
                  onClick={() => setZoom(100)}
                  style={{ border: "none", borderLeft: "1px solid #e2e8f0", background: "none", cursor: "pointer", padding: "3px 6px", fontSize: "10px", color: "#64748b" }}
                >
                  1:1
                </button>
              </div>

              {/* Open Full in New Tab (Inline Streaming) */}
              <a
                href={fileUrl}
                target="_blank"
                rel="noreferrer"
                title="Open file in dedicated tab"
                style={{
                  color: "#4f46e5",
                  fontSize: "12px",
                  fontWeight: "600",
                  textDecoration: "none",
                  marginLeft: "4px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "2px",
                }}
              >
                Expand &nearr;
              </a>
            </div>
          </div>

          {/* Document Render Area */}
          <div
            style={{
              flex: 1,
              backgroundColor: "#f1f5f9",
              borderRadius: "8px",
              border: "1px solid #e2e8f0",
              overflow: "auto",
              display: "flex",
              justifyContent: "center",
              alignItems: "flex-start",
              padding: "16px",
            }}
          >
            {isPdf ? (
              <div
                style={{
                  width: `${zoom}%`,
                  transition: "width 0.2s ease",
                  minWidth: "100%",
                  height: "100%",
                }}
              >
                <iframe
                  src={`${fileUrl}#page=${currentPage}&zoom=${zoom}`}
                  title="Document Preview"
                  style={{
                    width: "100%",
                    height: "100%",
                    minHeight: "560px",
                    border: "none",
                    borderRadius: "4px",
                    backgroundColor: "#ffffff",
                    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
                  }}
                />
              </div>
            ) : (
              <div
                style={{
                  transform: `scale(${zoom / 100})`,
                  transformOrigin: "top center",
                  transition: "transform 0.2s ease",
                }}
              >
                <img
                  src={fileUrl}
                  alt={doc.filename}
                  style={{
                    maxWidth: "100%",
                    height: "auto",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                    borderRadius: "4px",
                    backgroundColor: "#ffffff",
                  }}
                />
              </div>
            )}
          </div>

          {/* Preview Footer */}
          <div
            style={{
              marginTop: "10px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "11px",
              color: "#94a3b8",
            }}
          >
            <span>{doc.filename}</span>
            <span>{doc.pages_processed ?? 1} of {totalPages} pages parsed</span>
          </div>
        </div>

        {/* ── RIGHT PANEL: Extracted Information & Verification ── */}
        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            padding: "20px 24px",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Tab Navigation Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              borderBottom: "1px solid #e2e8f0",
              paddingBottom: "12px",
              marginBottom: "18px",
              flexWrap: "wrap",
              gap: "10px",
            }}
          >
            {/* Tabs */}
            <div style={{ display: "flex", gap: "6px" }}>
              <button
                type="button"
                onClick={() => setActiveTab("edit")}
                style={{
                  padding: "6px 12px",
                  fontSize: "13px",
                  fontWeight: activeTab === "edit" ? "600" : "500",
                  backgroundColor: activeTab === "edit" ? "#eef2ff" : "transparent",
                  color: activeTab === "edit" ? "#4f46e5" : "#64748b",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                }}
              >
                Extracted Fields
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("compare")}
                style={{
                  padding: "6px 12px",
                  fontSize: "13px",
                  fontWeight: activeTab === "compare" ? "600" : "500",
                  backgroundColor: activeTab === "compare" ? "#eef2ff" : "transparent",
                  color: activeTab === "compare" ? "#4f46e5" : "#64748b",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                }}
              >
                Method Comparison
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("audit")}
                style={{
                  padding: "6px 12px",
                  fontSize: "13px",
                  fontWeight: activeTab === "audit" ? "600" : "500",
                  backgroundColor: activeTab === "audit" ? "#eef2ff" : "transparent",
                  color: activeTab === "audit" ? "#4f46e5" : "#64748b",
                  border: "none",
                  borderRadius: "6px",
                  cursor: "pointer",
                }}
              >
                Audit Log ({auditLog.length})
              </button>
            </div>

            {/* Active Extraction Method Chip */}
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              {isLayoutLM ? (
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: "600",
                    padding: "3px 8px",
                    borderRadius: "6px",
                    backgroundColor: "#f5f3ff",
                    color: "#6d28d9",
                    border: "1px solid #ddd6fe",
                  }}
                >
                  ⚡ LayoutLMv3 Active
                </span>
              ) : (
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: "600",
                    padding: "3px 8px",
                    borderRadius: "6px",
                    backgroundColor: "#f1f5f9",
                    color: "#475569",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  📄 Baseline OCR Active
                </span>
              )}
            </div>
          </div>

          {/* TAB 1: EDITABLE FIELDS */}
          {activeTab === "edit" && (
            <div>
              <p style={{ color: "#64748b", fontSize: "12px", margin: "0 0 16px" }}>
                Verify and edit the canonical invoice fields below. Changes record audit timestamps and update database records immediately upon saving.
              </p>

              {/* SECTION A: Vendor and Customer Details */}
              <div style={{ marginBottom: "22px" }}>
                <div style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#475569", marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: "3px", height: "12px", backgroundColor: "#4f46e5", borderRadius: "2px" }} />
                  Vendor and Customer Details
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                  <FieldInput
                    label="Vendor Name"
                    value={fields.vendor_name}
                    onChange={(v) => handleFieldChange("vendor_name", v)}
                    confidence={confidenceMap.vendor_name}
                  />

                  <FieldInput
                    label="Customer Name"
                    value={fields.customer_name}
                    onChange={(v) => handleFieldChange("customer_name", v)}
                    confidence={confidenceMap.customer_name}
                  />

                  <FieldInput
                    label="Vendor GSTIN"
                    value={fields.gst_number}
                    onChange={(v) => handleFieldChange("gst_number", v)}
                    confidence={confidenceMap.gst_number}
                  />

                  <FieldInput
                    label="Customer GSTIN"
                    value={fields.customer_gst}
                    onChange={(v) => handleFieldChange("customer_gst", v)}
                    confidence={confidenceMap.customer_gst}
                  />
                </div>
              </div>

              {/* SECTION B: Invoice Information */}
              <div style={{ marginBottom: "22px" }}>
                <div style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#475569", marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: "3px", height: "12px", backgroundColor: "#2563eb", borderRadius: "2px" }} />
                  Invoice Information
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "12px" }}>
                  <FieldInput
                    label="Invoice Number"
                    value={fields.invoice_number}
                    onChange={(v) => handleFieldChange("invoice_number", v)}
                    confidence={confidenceMap.invoice_number}
                  />

                  <FieldInput
                    label="Invoice Date"
                    value={fields.invoice_date}
                    onChange={(v) => handleFieldChange("invoice_date", v)}
                    confidence={confidenceMap.invoice_date}
                    dateMeta={dateValidation.invoice_date}
                  />

                  <FieldInput
                    label="Due Date"
                    value={fields.due_date}
                    onChange={(v) => handleFieldChange("due_date", v)}
                    confidence={confidenceMap.due_date}
                    dateMeta={dateValidation.due_date}
                  />

                  <FieldInput
                    label="Currency"
                    value={fields.currency}
                    onChange={(v) => handleFieldChange("currency", v)}
                    confidence={confidenceMap.currency}
                  />
                </div>
              </div>

              {/* SECTION C: Financial Information */}
              <div style={{ marginBottom: "20px" }}>
                <div style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#475569", marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: "3px", height: "12px", backgroundColor: "#16a34a", borderRadius: "2px" }} />
                  Financial Breakdown
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
                  <FieldInput
                    label="Subtotal Amount"
                    value={fields.subtotal}
                    onChange={(v) => handleFieldChange("subtotal", v)}
                    confidence={confidenceMap.subtotal}
                  />

                  <FieldInput
                    label="CGST Amount"
                    value={fields.cgst_amount}
                    onChange={(v) => handleFieldChange("cgst_amount", v)}
                    confidence={confidenceMap.cgst_amount}
                  />

                  <FieldInput
                    label="SGST Amount"
                    value={fields.sgst_amount}
                    onChange={(v) => handleFieldChange("sgst_amount", v)}
                    confidence={confidenceMap.sgst_amount}
                  />

                  <FieldInput
                    label="Total Tax Amount"
                    value={fields.tax_amount}
                    onChange={(v) => handleFieldChange("tax_amount", v)}
                    confidence={confidenceMap.tax_amount}
                  />

                  <FieldInput
                    label="Total / Net Amount"
                    value={fields.total_amount}
                    onChange={(v) => handleFieldChange("total_amount", v)}
                    confidence={confidenceMap.total_amount}
                    isKeyTotal
                  />

                  <FieldInput
                    label="Gross Amount"
                    value={fields.gross_amount}
                    onChange={(v) => handleFieldChange("gross_amount", v)}
                    confidence={confidenceMap.gross_amount}
                  />
                </div>
              </div>

              {/* SECTION D: Financial Validation & Arithmetic Integrity */}
              <FinancialValidationCard
                validationResults={validationResults}
                fields={fields}
              />

              {/* SECTION E: Structured Line Items Table */}
              <LineItemsTable
                lineItems={lineItems}
                currency={fields.currency || "₹"}
              />

              {/* Confidence Notice */}
              <div
                style={{
                  padding: "10px 14px",
                  backgroundColor: "#f8fafc",
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  fontSize: "11px",
                  color: "#64748b",
                  lineHeight: 1.5,
                  marginBottom: "20px",
                }}
              >
                <strong style={{ color: "#334155" }}>Confidence Disclaimer: </strong>
                Confidence scores indicate model token probability during inference, not guaranteed factual accuracy. Always verify numbers with the original document image.
              </div>

              {/* Bottom Action Footer */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  paddingTop: "16px",
                  borderTop: "1px solid #e2e8f0",
                  flexWrap: "wrap",
                  gap: "10px",
                }}
              >
                <button
                  type="button"
                  onClick={handleSaveGroundTruth}
                  disabled={saving}
                  style={{
                    padding: "8px 14px",
                    fontSize: "12px",
                    fontWeight: "600",
                    backgroundColor: "#f1f5f9",
                    color: "#334155",
                    border: "1px solid #cbd5e1",
                    borderRadius: "6px",
                    cursor: saving ? "not-allowed" : "pointer",
                  }}
                >
                  Save as Benchmark Ground Truth
                </button>

                <div style={{ display: "flex", gap: "10px" }}>
                  <button
                    type="button"
                    onClick={handleSaveOnly}
                    disabled={saving || !hasUnsavedChanges}
                    style={{
                      padding: "8px 16px",
                      fontSize: "12px",
                      fontWeight: "600",
                      backgroundColor: "#ffffff",
                      color: hasUnsavedChanges ? "#0f172a" : "#94a3b8",
                      border: "1px solid #cbd5e1",
                      borderRadius: "6px",
                      cursor: hasUnsavedChanges ? "pointer" : "default",
                    }}
                  >
                    Save Changes
                  </button>

                  <button
                    type="button"
                    onClick={handleApprove}
                    disabled={saving}
                    style={{
                      padding: "8px 18px",
                      fontSize: "12px",
                      fontWeight: "600",
                      backgroundColor: "var(--primary, #4f46e5)",
                      color: "#ffffff",
                      border: "none",
                      borderRadius: "6px",
                      cursor: saving ? "not-allowed" : "pointer",
                      boxShadow: "0 1px 3px rgba(79, 70, 229, 0.25)",
                    }}
                  >
                    {verified ? "✓ Update Verified Record" : "✓ Approve & Verify"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: EXTRACTION COMPARISON (BASELINE VS LAYOUTLMV3) */}
          {activeTab === "compare" && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                <div>
                  <h3 style={{ fontSize: "14px", fontWeight: "700", color: "#0f172a", margin: "0 0 2px" }}>
                    Extraction Method Comparison
                  </h3>
                  <p style={{ fontSize: "12px", color: "#64748b", margin: 0 }}>
                    Side-by-side comparison between Heuristic Baseline OCR (O1) and LayoutLMv3 Spatial Classifier (O2)
                  </p>
                </div>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
                  <thead>
                    <tr style={{ backgroundColor: "#f8fafc", borderBottom: "1px solid #e2e8f0", color: "#475569" }}>
                      <th style={{ padding: "10px 12px", fontWeight: "600" }}>Canonical Field</th>
                      <th style={{ padding: "10px 12px", fontWeight: "600" }}>Baseline OCR (O1)</th>
                      <th style={{ padding: "10px 12px", fontWeight: "600" }}>LayoutLMv3 (O2)</th>
                      <th style={{ padding: "10px 12px", fontWeight: "600" }}>Comparison Status</th>
                      <th style={{ padding: "10px 12px", fontWeight: "600", textAlign: "right" }}>Adopt Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonList.map((item) => {
                      const isMatch = item.status === "match";
                      const isMismatch = item.status === "mismatch";
                      const isLayoutOnly = item.status === "layoutlm_only";

                      return (
                        <tr
                          key={item.key}
                          style={{
                            borderBottom: "1px solid #f1f5f9",
                            backgroundColor: isMismatch ? "#fef2f2" : "transparent",
                          }}
                        >
                          {/* Label */}
                          <td style={{ padding: "10px 12px", fontWeight: "600", color: "#0f172a" }}>
                            {item.label}
                          </td>

                          {/* Baseline Value */}
                          <td style={{ padding: "10px 12px", color: item.baselineValue ? "#334155" : "#94a3b8" }}>
                            {item.baselineValue || "—"}
                          </td>

                          {/* LayoutLMv3 Value */}
                          <td style={{ padding: "10px 12px", fontWeight: "500", color: item.layoutlmValue ? "#4c1d95" : "#94a3b8" }}>
                            {item.layoutlmValue || "—"}
                          </td>

                          {/* Status Pill */}
                          <td style={{ padding: "10px 12px" }}>
                            {isMatch ? (
                              <span style={{ fontSize: "11px", fontWeight: "600", color: "#16a34a", backgroundColor: "#ecfdf5", padding: "2px 6px", borderRadius: "4px" }}>
                                ✓ Match
                              </span>
                            ) : isMismatch ? (
                              <span style={{ fontSize: "11px", fontWeight: "600", color: "#dc2626", backgroundColor: "#fee2e2", padding: "2px 6px", borderRadius: "4px" }}>
                                ⚠ Mismatch
                              </span>
                            ) : isLayoutOnly ? (
                              <span style={{ fontSize: "11px", fontWeight: "600", color: "#6d28d9", backgroundColor: "#f5f3ff", padding: "2px 6px", borderRadius: "4px" }}>
                                LayoutLM Only
                              </span>
                            ) : (
                              <span style={{ fontSize: "11px", color: "#94a3b8" }}>
                                Both Empty
                              </span>
                            )}
                          </td>

                          {/* Adopt Action */}
                          <td style={{ padding: "10px 12px", textAlign: "right", whiteSpace: "nowrap" }}>
                            {item.layoutlmValue && (
                              <button
                                type="button"
                                onClick={() => handleFieldChange(item.key, item.layoutlmValue)}
                                style={{
                                  padding: "3px 8px",
                                  fontSize: "11px",
                                  backgroundColor: "#f5f3ff",
                                  color: "#6d28d9",
                                  border: "1px solid #ddd6fe",
                                  borderRadius: "4px",
                                  cursor: "pointer",
                                }}
                              >
                                Use LayoutLM
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Line-Item Extraction Capability Note */}
              <div
                style={{
                  marginTop: "20px",
                  padding: "14px 16px",
                  backgroundColor: "#f8fafc",
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: "700", color: "#334155", marginBottom: "4px" }}>
                  <TableIcon size={14} />
                  <span>Line-Item Table Extraction Comparison</span>
                </div>
                <p style={{ margin: 0, lineHeight: 1.5 }}>
                  Structured line-item row extraction is exclusively supported by <strong>LayoutLMv3 (O2)</strong> via multimodal 2D spatial clustering ({lineItems.length} table row{lineItems.length === 1 ? "" : "s"} extracted with arithmetic verification). Baseline OCR (O1) does not perform table row extraction, so side-by-side line-item comparison is omitted per system specification.
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: HUMAN-IN-THE-LOOP AUDIT TRAIL */}
          {activeTab === "audit" && (
            <div>
              <div style={{ marginBottom: "14px" }}>
                <h3 style={{ fontSize: "14px", fontWeight: "700", color: "#0f172a", margin: "0 0 2px" }}>
                  Persistent Audit Trail
                </h3>
                <p style={{ fontSize: "12px", color: "#64748b", margin: 0 }}>
                  Immutable log of all human edits, approvals, and field revisions for compliance
                </p>
              </div>

              {auditLog.length === 0 ? (
                <div style={{ padding: "32px 16px", textAlign: "center", color: "#94a3b8", fontSize: "13px" }}>
                  No modifications logged yet. Edits made in the review panel will be tracked here.
                </div>
              ) : (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
                    <thead>
                      <tr style={{ backgroundColor: "#f8fafc", borderBottom: "1px solid #e2e8f0", color: "#475569" }}>
                        <th style={{ padding: "8px 10px", fontWeight: "600" }}>Timestamp</th>
                        <th style={{ padding: "8px 10px", fontWeight: "600" }}>Action</th>
                        <th style={{ padding: "8px 10px", fontWeight: "600" }}>Field</th>
                        <th style={{ padding: "8px 10px", fontWeight: "600" }}>Old Value</th>
                        <th style={{ padding: "8px 10px", fontWeight: "600" }}>New Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {auditLog.map((entry) => (
                        <tr key={entry.id} style={{ borderBottom: "1px solid #f1f5f9" }}>
                          <td style={{ padding: "8px 10px", color: "#64748b", whiteSpace: "nowrap" }}>
                            {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "-"}
                          </td>
                          <td style={{ padding: "8px 10px" }}>
                            <span
                              style={{
                                padding: "2px 6px",
                                borderRadius: "4px",
                                fontSize: "10px",
                                fontWeight: "600",
                                textTransform: "uppercase",
                                backgroundColor: entry.action === "approve" ? "#dcfce7" : "#eff6ff",
                                color: entry.action === "approve" ? "#15803d" : "#2563eb",
                              }}
                            >
                              {entry.action}
                            </span>
                          </td>
                          <td style={{ padding: "8px 10px", fontWeight: "600", color: "#0f172a" }}>
                            {entry.field_name}
                          </td>
                          <td style={{ padding: "8px 10px", color: "#94a3b8" }}>{entry.old_value || "(empty)"}</td>
                          <td style={{ padding: "8px 10px", color: "#16a34a", fontWeight: "500" }}>{entry.new_value || "(empty)"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Reusable Component: Field Input with Confidence Indicator ────────
function FieldInput({ label, value, onChange, confidence, isKeyTotal = false, dateMeta = null }) {
  let confBadge;

  if (typeof confidence === "number" && !isNaN(confidence)) {
    // Normalise to percentage 0-100
    const confVal = confidence <= 1.0 ? Math.round(confidence * 100) : Math.round(confidence);

    if (confVal >= 80) {
      confBadge = { text: `${confVal}%`, level: "high", color: "#16a34a", bg: "#ecfdf5", border: "#a7f3d0" };
    } else if (confVal >= 50) {
      confBadge = { text: `${confVal}%`, level: "medium", color: "#b45309", bg: "#fffbeb", border: "#fde68a" };
    } else {
      confBadge = { text: `${confVal}% Needs Review`, level: "low", color: "#b91c1c", bg: "#fef2f2", border: "#fecaca" };
    }
  } else {
    confBadge = { text: "— N/A", level: "missing", color: "#94a3b8", bg: "#f8fafc", border: "#e2e8f0" };
  }

  const needsAttention = confBadge.level === "low" || (!value && isKeyTotal);

  return (
    <div style={{ marginBottom: "6px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "3px" }}>
        <label style={{ fontSize: "11px", fontWeight: "600", color: "#334155" }}>
          {label}
        </label>
        <span
          title="Inference token confidence score. Does not guarantee factual accuracy."
          style={{
            fontSize: "10px",
            fontWeight: "600",
            padding: "1px 5px",
            borderRadius: "4px",
            backgroundColor: confBadge.bg,
            color: confBadge.color,
            border: `1px solid ${confBadge.border}`,
          }}
        >
          {confBadge.text}
        </span>
      </div>

      <input
        type="text"
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Not detected"
        style={{
          width: "100%",
          padding: "7px 10px",
          border: `1px solid ${needsAttention ? "#f87171" : "#cbd5e1"}`,
          borderRadius: "6px",
          fontSize: "13px",
          fontWeight: isKeyTotal ? "700" : "500",
          backgroundColor: needsAttention ? "#fffbfa" : "#ffffff",
          color: "#0f172a",
          outline: "none",
          boxSizing: "border-box",
          transition: "border-color 0.15s ease",
        }}
      />

      {/* Date Normalization Details & Ambiguity / Error Warning */}
      {dateMeta && (
        <div style={{ marginTop: "4px", fontSize: "11px", lineHeight: 1.4 }}>
          {dateMeta.is_ambiguous ? (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "2px 7px",
                borderRadius: "4px",
                backgroundColor: "#fffbeb",
                color: "#b45309",
                border: "1px solid #fde68a",
                fontWeight: "600",
              }}
              title={dateMeta.error || "Format ambiguous between DD/MM and MM/DD"}
            >
              <AlertCircleIcon size={12} color="#b45309" />
              <span>
                Ambiguous date format
                {dateMeta.possible_dates && dateMeta.possible_dates.length > 0
                  ? ` (Could be ${dateMeta.possible_dates.join(" or ")})`
                  : ""}
                {dateMeta.raw ? ` [Raw: "${dateMeta.raw}"]` : ""}
              </span>
            </div>
          ) : dateMeta.is_valid === false && dateMeta.error ? (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "2px 7px",
                borderRadius: "4px",
                backgroundColor: "#fef2f2",
                color: "#dc2626",
                border: "1px solid #fecaca",
                fontWeight: "600",
              }}
            >
              <AlertCircleIcon size={12} color="#dc2626" />
              <span>{dateMeta.error} {dateMeta.raw ? `[Raw: "${dateMeta.raw}"]` : ""}</span>
            </div>
          ) : dateMeta.normalized ? (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "2px 6px",
                borderRadius: "4px",
                backgroundColor: "#ecfdf5",
                color: "#166534",
                border: "1px solid #a7f3d0",
                fontSize: "10px",
                fontWeight: "600",
              }}
            >
              <CheckCircleIcon size={11} color="#166534" />
              <span>ISO-8601: {dateMeta.normalized}</span>
              {dateMeta.raw && dateMeta.raw !== dateMeta.normalized && (
                <span style={{ color: "#6b7280", fontWeight: "normal" }}>({dateMeta.raw})</span>
              )}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

// ── Financial Validation Card Component ──────────────────────────────
function FinancialValidationCard({ validationResults, fields }) {
  const fin = validationResults?.financials || (validationResults?.calculated_total !== undefined ? validationResults : null);
  const currency = fields?.currency || "₹";

  const subtotal = fin?.subtotal !== null && fin?.subtotal !== undefined
    ? Number(fin.subtotal)
    : (fields?.subtotal ? parseFloat(String(fields.subtotal).replace(/[^0-9.-]/g, "")) : null);

  const tax = fin?.tax_amount !== null && fin?.tax_amount !== undefined
    ? Number(fin.tax_amount)
    : (fields?.tax_amount ? parseFloat(String(fields.tax_amount).replace(/[^0-9.-]/g, "")) : null);

  const discount = fin?.discount !== null && fin?.discount !== undefined ? Number(fin.discount) : 0;
  const shipping = fin?.shipping_amount !== null && fin?.shipping_amount !== undefined ? Number(fin.shipping_amount) : 0;

  const extractedTotal = fin?.extracted_total !== null && fin?.extracted_total !== undefined
    ? Number(fin.extracted_total)
    : (fields?.total_amount ? parseFloat(String(fields.total_amount).replace(/[^0-9.-]/g, "")) : null);

  const calculatedTotal = fin?.calculated_total !== null && fin?.calculated_total !== undefined
    ? Number(fin.calculated_total)
    : (subtotal !== null ? Math.round((subtotal + (tax || 0) - discount + shipping) * 100) / 100 : null);

  const diff = fin?.difference !== null && fin?.difference !== undefined
    ? Number(fin.difference)
    : (calculatedTotal !== null && extractedTotal !== null
        ? Math.round(Math.abs(calculatedTotal - extractedTotal) * 100) / 100
        : null);

  const tolerance = fin?.tolerance || 0.05;
  const hasErrors = Boolean(fin?.validation_errors && fin.validation_errors.length > 0);
  const isReconciled = diff !== null ? (diff <= tolerance && !hasErrors && fin?.is_valid !== false) : (fin?.is_valid ?? null);

  const formatAmount = (val) => {
    if (val === null || val === undefined || isNaN(val)) return "—";
    return `${currency} ${Number(val).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        borderRadius: "10px",
        border: `1px solid ${isReconciled === true ? "#a7f3d0" : isReconciled === false ? "#fecaca" : "#e2e8f0"}`,
        padding: "16px 18px",
        marginBottom: "22px",
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: "14px",
          paddingBottom: "10px",
          borderBottom: "1px solid #f1f5f9",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div
            style={{
              width: "28px",
              height: "28px",
              borderRadius: "6px",
              backgroundColor: isReconciled === true ? "#ecfdf5" : isReconciled === false ? "#fef2f2" : "#f1f5f9",
              color: isReconciled === true ? "#16a34a" : isReconciled === false ? "#dc2626" : "#475569",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <CalculatorIcon size={16} />
          </div>
          <div>
            <h4 style={{ fontSize: "13px", fontWeight: "700", color: "#0f172a", margin: 0 }}>
              Financial Arithmetic Validation
            </h4>
            <span style={{ fontSize: "11px", color: "#64748b" }}>
              Mathematical cross-check: Subtotal + Tax - Discount + Additional Charges vs Extracted Total
            </span>
          </div>
        </div>

        {/* Status Pill */}
        {isReconciled === true ? (
          <span
            style={{
              fontSize: "11px",
              fontWeight: "600",
              padding: "3px 10px",
              borderRadius: "12px",
              backgroundColor: "#ecfdf5",
              color: "#15803d",
              border: "1px solid #a7f3d0",
              display: "inline-flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <CheckCircleIcon size={12} color="#15803d" />
            Reconciled (Consistent)
          </span>
        ) : isReconciled === false ? (
          <span
            style={{
              fontSize: "11px",
              fontWeight: "600",
              padding: "3px 10px",
              borderRadius: "12px",
              backgroundColor: "#fef2f2",
              color: "#b91c1c",
              border: "1px solid #fecaca",
              display: "inline-flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            <AlertCircleIcon size={12} color="#b91c1c" />
            Discrepancy Detected
          </span>
        ) : (
          <span
            style={{
              fontSize: "11px",
              fontWeight: "600",
              padding: "3px 10px",
              borderRadius: "12px",
              backgroundColor: "#f8fafc",
              color: "#64748b",
              border: "1px solid #e2e8f0",
            }}
          >
            Pending Analysis
          </span>
        )}
      </div>

      {/* Grid of Financial Variables */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
          gap: "10px",
          marginBottom: "12px",
        }}
      >
        <FinancialStatBox label="Subtotal" value={formatAmount(subtotal)} />
        <FinancialStatBox label="+ Tax Amount" value={formatAmount(tax)} />
        {discount > 0 && <FinancialStatBox label="- Discount" value={formatAmount(discount)} />}
        {shipping > 0 && <FinancialStatBox label="+ Additional Charges" value={formatAmount(shipping)} />}
        <FinancialStatBox
          label="Calculated Total"
          value={formatAmount(calculatedTotal)}
          highlight={isReconciled === true}
        />
        <FinancialStatBox
          label="Extracted Total"
          value={formatAmount(extractedTotal)}
          isKeyTotal
        />
        <FinancialStatBox
          label="Difference"
          value={diff !== null ? formatAmount(diff) : "—"}
          isDiscrepancy={diff !== null && diff > tolerance}
        />
        {fin?.line_items_sum !== null && fin?.line_items_sum !== undefined && (
          <FinancialStatBox
            label="Line Items Sum"
            value={formatAmount(fin.line_items_sum)}
            subtext={fin.line_items_consistent ? "✓ Matches Subtotal" : "⚠ Diff vs Subtotal"}
            subtextColor={fin.line_items_consistent ? "#16a34a" : "#dc2626"}
          />
        )}
      </div>

      {/* Validation Errors Box */}
      {hasErrors && (
        <div
          style={{
            backgroundColor: "#fef2f2",
            borderRadius: "6px",
            border: "1px solid #fecaca",
            padding: "8px 12px",
            marginTop: "10px",
          }}
        >
          <div style={{ fontSize: "11px", fontWeight: "700", color: "#b91c1c", marginBottom: "4px" }}>
            Validation Warnings & Discrepancies:
          </div>
          <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "11px", color: "#991b1b" }}>
            {fin.validation_errors.map((err, i) => (
              <li key={i} style={{ marginBottom: "2px" }}>{err}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Financial Stat Box Helper ────────────────────────────────────────
function FinancialStatBox({
  label,
  value,
  isKeyTotal = false,
  isDiscrepancy = false,
  highlight = false,
  subtext = null,
  subtextColor = "#64748b",
}) {
  return (
    <div
      style={{
        padding: "8px 10px",
        backgroundColor: isDiscrepancy ? "#fef2f2" : highlight ? "#f0fdf4" : isKeyTotal ? "#f8fafc" : "#ffffff",
        border: `1px solid ${isDiscrepancy ? "#fecaca" : highlight ? "#bbf7d0" : isKeyTotal ? "#cbd5e1" : "#e2e8f0"}`,
        borderRadius: "6px",
      }}
    >
      <div style={{ fontSize: "10px", fontWeight: "600", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: "2px" }}>
        {label}
      </div>
      <div
        style={{
          fontSize: isKeyTotal ? "14px" : "13px",
          fontWeight: isKeyTotal || highlight ? "700" : "600",
          color: isDiscrepancy ? "#dc2626" : highlight ? "#15803d" : "#0f172a",
        }}
      >
        {value}
      </div>
      {subtext && (
        <div style={{ fontSize: "10px", color: subtextColor, marginTop: "2px", fontWeight: "600" }}>
          {subtext}
        </div>
      )}
    </div>
  );
}

// ── Structured Line Items Table Component ────────────────────────────
function LineItemsTable({ lineItems = [], currency = "₹" }) {
  const items = useMemo(() => (Array.isArray(lineItems) ? lineItems : []), [lineItems]);

  const formatNum = (val) => {
    if (val === null || val === undefined || val === "") return "—";
    const num = Number(val);
    return isNaN(num) ? String(val) : num.toLocaleString();
  };

  const formatPrice = (val) => {
    if (val === null || val === undefined || val === "") return "—";
    const num = Number(val);
    return isNaN(num) ? String(val) : `${currency} ${num.toFixed(2)}`;
  };

  const totalSum = useMemo(() => {
    return items.reduce((acc, it) => {
      const val = Number(it.item_total);
      return !isNaN(val) ? acc + val : acc;
    }, 0);
  }, [items]);

  return (
    <div style={{ marginBottom: "24px" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "12px",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#475569", display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: "3px", height: "12px", backgroundColor: "#0284c7", borderRadius: "2px" }} />
          Structured Line Items ({items.length})
        </div>

        <span
          style={{
            fontSize: "10px",
            fontWeight: "600",
            padding: "2px 8px",
            borderRadius: "6px",
            backgroundColor: "#f1f5f9",
            color: "#64748b",
            border: "1px solid #e2e8f0",
          }}
        >
          Read-only (Extracted from layout)
        </span>
      </div>

      {items.length === 0 ? (
        <div
          style={{
            padding: "24px 16px",
            textAlign: "center",
            backgroundColor: "#f8fafc",
            borderRadius: "8px",
            border: "1px dashed #cbd5e1",
            color: "#64748b",
            fontSize: "12px",
          }}
        >
          <div style={{ width: "32px", height: "32px", borderRadius: "50%", backgroundColor: "#e2e8f0", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 8px", color: "#94a3b8" }}>
            <TableIcon size={16} />
          </div>
          <strong style={{ color: "#334155" }}>No line items detected</strong>
          <p style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: "11px" }}>
            Structured line items will be populated if the invoice contains table rows and LayoutLMv3 is run.
          </p>
        </div>
      ) : (
        <div
          style={{
            overflowX: "auto",
            backgroundColor: "#ffffff",
            borderRadius: "8px",
            border: "1px solid #e2e8f0",
            boxShadow: "0 1px 2px rgba(15, 23, 42, 0.03)",
          }}
        >
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
            <thead>
              <tr style={{ backgroundColor: "#f8fafc", borderBottom: "1px solid #e2e8f0", color: "#475569" }}>
                <th style={{ padding: "8px 10px", fontWeight: "600", width: "40px" }}>#</th>
                <th style={{ padding: "8px 10px", fontWeight: "600", width: "50px" }}>Page</th>
                <th style={{ padding: "8px 10px", fontWeight: "600" }}>Description</th>
                <th style={{ padding: "8px 10px", fontWeight: "600", textAlign: "right", width: "70px" }}>Qty</th>
                <th style={{ padding: "8px 10px", fontWeight: "600", textAlign: "right", width: "95px" }}>Unit Price</th>
                <th style={{ padding: "8px 10px", fontWeight: "600", textAlign: "right", width: "95px" }}>Total</th>
                <th style={{ padding: "8px 10px", fontWeight: "600", textAlign: "center", width: "105px" }}>Validation</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it, idx) => {
                const rowIndex = it.row_index ?? idx + 1;
                const pageNum = it.page ?? 1;
                const desc = it.description || it.raw_description || "—";
                const qty = formatNum(it.quantity ?? it.raw_quantity);
                const unitPrice = formatPrice(it.unit_price ?? it.raw_unit_price);
                const itemTotal = formatPrice(it.item_total ?? it.raw_item_total);

                const isValid = it.is_valid !== false && !it.suspicious && !it.validation_error;
                const isMismatch = it.suspicious || Boolean(it.validation_error);
                const isPartial = !isMismatch && (it.quantity === null || it.unit_price === null) && it.item_total !== null;

                return (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: "1px solid #f1f5f9",
                      backgroundColor: isMismatch ? "#fffbfa" : idx % 2 === 1 ? "#fafbfc" : "#ffffff",
                    }}
                  >
                    <td style={{ padding: "8px 10px", color: "#64748b", fontWeight: "500" }}>{rowIndex}</td>
                    <td style={{ padding: "8px 10px", color: "#64748b" }}>
                      <span style={{ padding: "1px 5px", borderRadius: "3px", backgroundColor: "#f1f5f9", fontSize: "10px", fontWeight: "600" }}>
                        P{pageNum}
                      </span>
                    </td>
                    <td style={{ padding: "8px 10px", color: "#0f172a", fontWeight: "500", wordBreak: "break-word" }}>
                      {desc}
                    </td>
                    <td style={{ padding: "8px 10px", textAlign: "right", color: "#334155" }}>{qty}</td>
                    <td style={{ padding: "8px 10px", textAlign: "right", color: "#334155" }}>{unitPrice}</td>
                    <td style={{ padding: "8px 10px", textAlign: "right", fontWeight: "600", color: "#0f172a" }}>
                      {itemTotal}
                    </td>
                    <td style={{ padding: "8px 10px", textAlign: "center" }}>
                      {isMismatch ? (
                        <span
                          title={it.validation_error || "Arithmetic discrepancy in row calculation"}
                          style={{
                            fontSize: "10px",
                            fontWeight: "600",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            backgroundColor: "#fee2e2",
                            color: "#dc2626",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "3px",
                          }}
                        >
                          <AlertCircleIcon size={10} color="#dc2626" />
                          Mismatch
                        </span>
                      ) : isPartial ? (
                        <span
                          title="Partial row components (e.g. total extracted without individual unit price)"
                          style={{
                            fontSize: "10px",
                            fontWeight: "600",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            backgroundColor: "#f1f5f9",
                            color: "#475569",
                          }}
                        >
                          Partial
                        </span>
                      ) : isValid ? (
                        <span
                          style={{
                            fontSize: "10px",
                            fontWeight: "600",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            backgroundColor: "#ecfdf5",
                            color: "#16a34a",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "3px",
                          }}
                        >
                          <CheckCircleIcon size={10} color="#16a34a" />
                          Valid
                        </span>
                      ) : (
                        <span style={{ fontSize: "10px", color: "#94a3b8" }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
            {items.length > 0 && totalSum > 0 && (
              <tfoot>
                <tr style={{ backgroundColor: "#f8fafc", borderTop: "2px solid #e2e8f0", fontWeight: "600" }}>
                  <td colSpan={3} style={{ padding: "8px 10px", color: "#475569" }}>
                    Total Line Items: {items.length} row{items.length === 1 ? "" : "s"}
                  </td>
                  <td colSpan={2} style={{ padding: "8px 10px", textAlign: "right", color: "#64748b" }}>
                    Sum of Line Totals:
                  </td>
                  <td style={{ padding: "8px 10px", textAlign: "right", color: "#0f172a", fontWeight: "700" }}>
                    {currency} {totalSum.toFixed(2)}
                  </td>
                  <td style={{ padding: "8px 10px" }} />
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      )}
    </div>
  );
}

// ── Vector SVG Icons ─────────────────────────────────────────────────
function FileTextIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function SparklesIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
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

function CheckCircleIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function AlertCircleIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

function TableIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M3 9h18" />
      <path d="M3 15h18" />
      <path d="M9 3v18" />
      <path d="M15 3v18" />
    </svg>
  );
}

function CalculatorIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="16" height="20" x="4" y="2" rx="2" />
      <line x1="8" x2="16" y1="6" y2="6" />
      <line x1="16" x2="16" y1="14" y2="18" />
      <path d="M16 10h.01" />
      <path d="M12 10h.01" />
      <path d="M8 10h.01" />
      <path d="M12 14h.01" />
      <path d="M8 14h.01" />
      <path d="M12 18h.01" />
      <path d="M8 18h.01" />
    </svg>
  );
}
