import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import { Link } from "react-router-dom";
import { getDocuments, getDocument, askQuestion, getDocumentFileUrl } from "../services/api";

// ── SVG Icon Helpers ─────────────────────────────────────────────────────────

function SparklesIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
    </svg>
  );
}

function UserIcon({ size = 15, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function SendIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function FileTextIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

function TrashIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function CopyIcon({ size = 13, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon({ size = 13, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AlertTriangleIcon({ size = 15, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function ExternalLinkIcon({ size = 13, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  );
}

function LightningIcon({ size = 13, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function HelpCircleIcon({ size = 15, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function RefreshIcon({ size = 14, spinning = false, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        animation: spinning ? "spin 0.8s linear infinite" : "none",
        transformOrigin: "center",
      }}
    >
      <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
    </svg>
  );
}

// ── Suggested Prompt Candidates ──────────────────────────────────────────────

const SUGGESTED_QUESTIONS = [
  "What is the invoice number?",
  "Who is the vendor?",
  "What is the invoice date?",
  "What is the total invoice amount?",
  "What are the CGST and SGST amounts?",
  "What products are listed in this invoice?",
  "What is the customer name?",
  "What is the vendor GSTIN?",
];

// ── Main Component ───────────────────────────────────────────────────────────

export default function DocumentQA() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState("");
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [question, setQuestion] = useState("");
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [asking, setAsking] = useState(false);
  const [chatHistories, setChatHistories] = useState({}); // mapped by docId
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("chat"); // 'chat' | 'preview'
  const [copiedAnswerId, setCopiedAnswerId] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Active document's conversation history
  const currentChatHistory = useMemo(() => {
    return selectedDocId ? chatHistories[selectedDocId] || [] : [];
  }, [chatHistories, selectedDocId]);

  // Initial fetch of document records
  useEffect(() => {
    let active = true;

    async function fetchDocuments() {
      try {
        const res = await getDocuments({ limit: 100 });
        const docs = res.documents || [];
        if (active) {
          setDocuments(docs);
          if (docs.length > 0) {
            setSelectedDocId(docs[0].id);
          }
        }
      } catch {
        if (active) {
          setError("Failed to load documents from backend. Please ensure the server is active.");
        }
      } finally {
        if (active) {
          setLoadingDocs(false);
        }
      }
    }

    fetchDocuments();

    return () => {
      active = false;
    };
  }, []);

  // Fetch full details of target document when selection changes
  useEffect(() => {
    if (!selectedDocId) return;

    let active = true;

    async function fetchDocDetails() {
      try {
        const docData = await getDocument(selectedDocId);
        if (active) {
          setSelectedDoc(docData);
        }
      } catch {
        // fallback to list metadata if individual lookup fails
      }
    }

    fetchDocDetails();

    return () => {
      active = false;
    };
  }, [selectedDocId]);

  // Auto-scroll chat to latest message
  useEffect(() => {
    if (activeTab === "chat") {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [currentChatHistory, asking, activeTab]);

  // Handle Question Submission
  const handleAsk = useCallback(async (queryText) => {
    const q = (queryText || question).trim();
    if (!q || !selectedDocId || asking) return;

    setAsking(true);
    setError("");

    const turnId = Date.now();
    const userTimestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    // Append user message immediately
    const userTurn = {
      id: turnId,
      sender: "user",
      text: q,
      timestamp: userTimestamp,
    };

    setChatHistories((prev) => ({
      ...prev,
      [selectedDocId]: [...(prev[selectedDocId] || []), userTurn],
    }));

    setQuestion("");

    try {
      const response = await askQuestion(selectedDocId, q);

      const aiTurn = {
        id: turnId + 1,
        sender: "ai",
        question: q,
        answer: response.answer,
        confidence: response.confidence,
        confidence_level: response.confidence_level,
        source_text: response.source_text,
        explanation: response.explanation,
        model: response.model || "impira/layoutlm-document-qa",
        source_type: response.source_type || null,
        field_name: response.field_name || null,
        page_number: response.page_number != null ? response.page_number : null,
        pages_searched: response.pages_searched != null ? response.pages_searched : null,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isError: false,
      };

      setChatHistories((prev) => ({
        ...prev,
        [selectedDocId]: [...(prev[selectedDocId] || []), aiTurn],
      }));
    } catch (err) {
      const errorTurn = {
        id: turnId + 1,
        sender: "ai",
        question: q,
        answer: null,
        error: err.message || "Failed to retrieve answer from Document QA model.",
        confidence: 0,
        confidence_level: "LOW",
        model: "impira/layoutlm-document-qa",
        page_number: null,
        pages_searched: null,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        isError: true,
      };

      setChatHistories((prev) => ({
        ...prev,
        [selectedDocId]: [...(prev[selectedDocId] || []), errorTurn],
      }));
    } finally {
      setAsking(false);
      // keep focus on input
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [question, selectedDocId, asking]);

  // Clear current document's conversation
  const handleClearChat = () => {
    if (!selectedDocId) return;
    setChatHistories((prev) => ({
      ...prev,
      [selectedDocId]: [],
    }));
  };

  // Copy answer text to clipboard
  const handleCopyAnswer = (answerText, turnId) => {
    if (!navigator.clipboard || !answerText) return;
    navigator.clipboard.writeText(answerText);
    setCopiedAnswerId(turnId);
    setTimeout(() => setCopiedAnswerId(null), 2000);
  };

  // Loading state
  if (loadingDocs) {
    return (
      <div style={{ maxWidth: "1280px", margin: "0 auto", padding: "60px 20px", textAlign: "center" }}>
        <div
          style={{
            width: "48px",
            height: "48px",
            borderRadius: "50%",
            border: "3px solid #e2e8f0",
            borderTopColor: "#6d28d9",
            animation: "spin 0.8s linear infinite",
            margin: "0 auto 16px",
          }}
        />
        <h2 style={{ fontSize: "18px", color: "#0f172a", margin: "0 0 6px" }}>Initializing Document QA Assistant</h2>
        <p style={{ color: "#64748b", fontSize: "14px", margin: 0 }}>
          Connecting to extractive visual language models and grounding pipeline...
        </p>
      </div>
    );
  }

  // Empty state: no documents uploaded yet
  if (documents.length === 0) {
    return (
      <div style={{ maxWidth: "1280px", margin: "0 auto", paddingBottom: "40px" }}>
        <div style={{ marginBottom: "24px" }}>
          <h1 style={{ fontSize: "26px", fontWeight: "700", color: "#0f172a", margin: "0 0 6px" }}>
            Document Question Answering
          </h1>
          <p style={{ color: "#64748b", fontSize: "14px", margin: 0 }}>
            Research Objective O3: Extractive machine reading comprehension over document visual layout.
          </p>
        </div>

        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: "14px",
            border: "1px solid #e2e8f0",
            padding: "50px 24px",
            textAlign: "center",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
          }}
        >
          <div
            style={{
              width: "56px",
              height: "56px",
              borderRadius: "14px",
              backgroundColor: "#f5f3ff",
              color: "#6d28d9",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "16px",
            }}
          >
            <SparklesIcon size={28} color="#6d28d9" />
          </div>
          <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#0f172a", margin: "0 0 8px" }}>
            No Ingested Documents Available
          </h2>
          <p style={{ color: "#64748b", fontSize: "14px", margin: "0 0 24px", maxWidth: "460px", marginInline: "auto" }}>
            The Document QA model requires an ingested invoice, receipt, or form image to evaluate spatial tokens and extract answers.
          </p>
          <Link
            to="/upload"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "10px 22px",
              backgroundColor: "#4f46e5",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: "600",
              boxShadow: "0 1px 3px rgba(79, 70, 229, 0.25)",
            }}
          >
            + Upload Document
          </Link>
        </div>
      </div>
    );
  }

  // Active document metadata derivation
  const isLayoutLM =
    selectedDoc?.extraction_method === "layoutlm" ||
    Boolean(selectedDoc?.layoutlm_fields && Object.keys(selectedDoc?.layoutlm_fields).length > 0);

  const activeFields =
    isLayoutLM && selectedDoc?.layoutlm_fields && Object.keys(selectedDoc?.layoutlm_fields).length > 0
      ? selectedDoc.layoutlm_fields
      : (selectedDoc?.invoice_fields || selectedDoc?.extracted_fields || {});

  const vendorName = activeFields.vendor_name || activeFields.company_name || null;
  const invoiceNumber = activeFields.invoice_number || null;
  const invoiceDate = activeFields.invoice_date || null;
  const rawTotal = activeFields.total_amount || activeFields.subtotal || null;
  const currency =
    activeFields.currency && activeFields.currency !== "Not found"
      ? activeFields.currency
      : rawTotal
      ? "₹"
      : "";
  const totalAmountFormatted = rawTotal && rawTotal !== "Not found" ? `${currency} ${rawTotal}` : null;
  const gstinNumber = activeFields.vendor_gstin || activeFields.gst_number || null;

  const isPdf = selectedDoc?.filename?.toLowerCase().endsWith(".pdf");
  const fileUrl = selectedDocId ? getDocumentFileUrl(selectedDocId) : null;

  return (
    <div style={{ maxWidth: "1360px", margin: "0 auto", paddingBottom: "48px" }}>
      {/* ── 1. Page Header ──────────────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "16px",
          marginBottom: "20px",
          paddingBottom: "18px",
          borderBottom: "1px solid #e2e8f0",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
            <h1
              style={{
                fontSize: "26px",
                fontWeight: "700",
                color: "#0f172a",
                margin: 0,
                letterSpacing: "-0.02em",
              }}
            >
              Document Question Answering
            </h1>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "3px 10px",
                fontSize: "11px",
                fontWeight: "600",
                borderRadius: "12px",
                backgroundColor: "#ede9fe",
                color: "#6d28d9",
                border: "1px solid #ddd6fe",
              }}
            >
              <SparklesIcon size={12} color="#6d28d9" />
              Objective O3
            </span>
            <span
              style={{
                padding: "3px 9px",
                fontSize: "11px",
                fontWeight: "600",
                borderRadius: "12px",
                backgroundColor: "#f1f5f9",
                color: "#475569",
                border: "1px solid #e2e8f0",
              }}
            >
              LayoutLM DocQA (Extractive)
            </span>
          </div>

          <p
            style={{
              color: "#475569",
              fontSize: "14px",
              margin: 0,
              maxWidth: "760px",
              lineHeight: 1.5,
            }}
          >
            Inquire directly about visual layout, spatial token spans, and invoice line items. Responses are grounded in actual document context using spatial transformer comprehension.
          </p>
        </div>

        {/* Top Header Actions */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <Link
            to="/history"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              backgroundColor: "#ffffff",
              color: "#475569",
              border: "1px solid #cbd5e1",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              textDecoration: "none",
            }}
          >
            <FileTextIcon size={14} color="#475569" />
            Document History
          </Link>
          <Link
            to={selectedDocId ? `/review/${selectedDocId}` : "/review"}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              backgroundColor: "#4f46e5",
              color: "#ffffff",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              textDecoration: "none",
            }}
          >
            Document Review &rarr;
          </Link>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          style={{
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            color: "#b91c1c",
            padding: "12px 18px",
            borderRadius: "8px",
            marginBottom: "20px",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <AlertTriangleIcon size={16} color="#b91c1c" />
          <span>{error}</span>
        </div>
      )}

      {/* ── 2. Document Selection Bar ────────────────────────────────────────── */}
      <div
        style={{
          backgroundColor: "#ffffff",
          padding: "16px 20px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
          marginBottom: "20px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "14px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: "1 1 340px" }}>
          <label
            htmlFor="target-doc-select"
            style={{
              fontSize: "13px",
              fontWeight: "700",
              color: "#334155",
              whiteSpace: "nowrap",
              textTransform: "uppercase",
              letterSpacing: "0.04em",
            }}
          >
            Target Document:
          </label>
          <select
            id="target-doc-select"
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            style={{
              flex: 1,
              padding: "9px 12px",
              border: "1px solid #cbd5e1",
              borderRadius: "8px",
              fontSize: "13px",
              color: "#0f172a",
              backgroundColor: "#ffffff",
              fontWeight: "600",
              outline: "none",
              cursor: "pointer",
            }}
          >
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename} — ({d.document_type || "Invoice"} • {d.coverage_score ?? d.confidence_score ?? 0}% coverage)
              </option>
            ))}
          </select>
        </div>

        {/* View Switcher Tabs (Chat vs Full Document Preview) */}
        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
          <button
            onClick={() => setActiveTab("chat")}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              border: "1px solid",
              borderColor: activeTab === "chat" ? "#4f46e5" : "#e2e8f0",
              backgroundColor: activeTab === "chat" ? "#eef2ff" : "#ffffff",
              color: activeTab === "chat" ? "#4f46e5" : "#64748b",
              cursor: "pointer",
              transition: "all 0.12s ease",
            }}
          >
            <SparklesIcon size={14} color={activeTab === "chat" ? "#4f46e5" : "#64748b"} />
            AI Chat Assistant
          </button>

          <button
            onClick={() => setActiveTab("preview")}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "8px 14px",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              border: "1px solid",
              borderColor: activeTab === "preview" ? "#4f46e5" : "#e2e8f0",
              backgroundColor: activeTab === "preview" ? "#eef2ff" : "#ffffff",
              color: activeTab === "preview" ? "#4f46e5" : "#64748b",
              cursor: "pointer",
              transition: "all 0.12s ease",
            }}
          >
            <FileTextIcon size={14} color={activeTab === "preview" ? "#4f46e5" : "#64748b"} />
            Document Preview
          </button>
        </div>
      </div>

      {/* ── 3. Main Workspace Layout ────────────────────────────────────────── */}
      <div
        className={activeTab === "preview" ? "" : "qa-workspace-grid"}
        style={{
          display: activeTab === "preview" ? "block" : undefined,
          gap: "20px",
          alignItems: "start",
        }}
      >
        {/* Left Column: Compact Document Context Panel (visible in chat mode) */}
        {activeTab === "chat" && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            {/* Document Context Card */}
            <div
              style={{
                backgroundColor: "#ffffff",
                borderRadius: "12px",
                border: "1px solid #e2e8f0",
                boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
                padding: "18px 20px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  marginBottom: "12px",
                  paddingBottom: "10px",
                  borderBottom: "1px solid #f1f5f9",
                }}
              >
                <div>
                  <span style={{ fontSize: "11px", fontWeight: "700", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Active Context
                  </span>
                  <h3
                    style={{
                      fontSize: "15px",
                      fontWeight: "700",
                      color: "#0f172a",
                      margin: "4px 0 0",
                      wordBreak: "break-all",
                    }}
                  >
                    {selectedDoc?.filename}
                  </h3>
                </div>

                {isLayoutLM ? (
                  <span
                    style={{
                      padding: "2px 8px",
                      fontSize: "11px",
                      fontWeight: "600",
                      borderRadius: "10px",
                      backgroundColor: "#f5f3ff",
                      color: "#6d28d9",
                      border: "1px solid #ddd6fe",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "3px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <LightningIcon size={11} color="#6d28d9" />
                    LayoutLM
                  </span>
                ) : (
                  <span
                    style={{
                      padding: "2px 8px",
                      fontSize: "11px",
                      fontWeight: "600",
                      borderRadius: "10px",
                      backgroundColor: "#f1f5f9",
                      color: "#475569",
                      border: "1px solid #e2e8f0",
                      whiteSpace: "nowrap",
                    }}
                  >
                    Baseline
                  </span>
                )}
              </div>

              {/* Context Key Fields List */}
              <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "12px" }}>
                <ContextFieldRow
                  label="Document Type"
                  value={selectedDoc?.document_type ? selectedDoc.document_type.toUpperCase() : "INVOICE"}
                />
                <ContextFieldRow
                  label="Vendor Name"
                  value={vendorName}
                  highlight={Boolean(vendorName)}
                />
                <ContextFieldRow
                  label="Invoice Number"
                  value={invoiceNumber}
                  isMono={Boolean(invoiceNumber)}
                />
                <ContextFieldRow
                  label="Invoice Date"
                  value={invoiceDate}
                />
                <ContextFieldRow
                  label="Total Amount"
                  value={totalAmountFormatted}
                  highlight={Boolean(totalAmountFormatted)}
                />
                <ContextFieldRow
                  label="GSTIN"
                  value={gstinNumber}
                  isMono={Boolean(gstinNumber)}
                />
                <ContextFieldRow
                  label="Verification"
                  value={selectedDoc?.verified ? "✓ Verified" : "⚠ Pending Review"}
                  textColor={selectedDoc?.verified ? "#15803d" : "#c2410c"}
                />
                <ContextFieldRow
                  label="Page Count"
                  value={selectedDoc?.page_count ? `${selectedDoc.page_count} page(s)` : "1 page"}
                />
              </div>

              {/* Document Actions Footer */}
              <div
                style={{
                  marginTop: "16px",
                  paddingTop: "12px",
                  borderTop: "1px solid #f1f5f9",
                  display: "flex",
                  gap: "8px",
                }}
              >
                <a
                  href={fileUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    flex: 1,
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "5px",
                    padding: "7px 10px",
                    fontSize: "12px",
                    fontWeight: "600",
                    color: "#334155",
                    backgroundColor: "#f8fafc",
                    border: "1px solid #cbd5e1",
                    borderRadius: "6px",
                    textDecoration: "none",
                  }}
                >
                  <ExternalLinkIcon size={12} />
                  Open File
                </a>

                <Link
                  to={`/review/${selectedDocId}`}
                  style={{
                    flex: 1,
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "5px",
                    padding: "7px 10px",
                    fontSize: "12px",
                    fontWeight: "600",
                    color: "#4f46e5",
                    backgroundColor: "#eef2ff",
                    border: "1px solid #c7d2fe",
                    borderRadius: "6px",
                    textDecoration: "none",
                  }}
                >
                  Review Page
                </Link>
              </div>
            </div>

            {/* Model Grounding Specs Card */}
            <div
              style={{
                backgroundColor: "#f8fafc",
                borderRadius: "12px",
                border: "1px solid #e2e8f0",
                padding: "16px 18px",
                fontSize: "12px",
                color: "#64748b",
                lineHeight: 1.5,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#334155", fontWeight: "700", marginBottom: "6px" }}>
                <HelpCircleIcon size={14} color="#6d28d9" />
                <span>Extractive QA Architecture</span>
              </div>
              <p style={{ margin: "0 0 8px" }}>
                Answers are extracted directly from the document token stream without generative hallucinations.
              </p>
              <div style={{ fontSize: "11px", color: "#94a3b8" }}>
                Checkpoint: <code>impira/layoutlm-document-qa</code>
              </div>
            </div>
          </div>
        )}

        {/* Full Document Preview Tab (if activeTab === 'preview') */}
        {activeTab === "preview" && (
          <div
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "12px",
              border: "1px solid #e2e8f0",
              boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                padding: "14px 20px",
                borderBottom: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                backgroundColor: "#f8fafc",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <FileTextIcon size={16} color="#4f46e5" />
                <span style={{ fontWeight: "700", color: "#0f172a", fontSize: "14px" }}>
                  {selectedDoc?.filename}
                </span>
              </div>

              <a
                href={fileUrl}
                target="_blank"
                rel="noreferrer"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                  fontSize: "12px",
                  color: "#4f46e5",
                  fontWeight: "600",
                  textDecoration: "none",
                }}
              >
                <span>Open in Tab</span>
                <ExternalLinkIcon size={12} />
              </a>
            </div>

            <div
              style={{
                minHeight: "600px",
                backgroundColor: "#f1f5f9",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "16px",
              }}
            >
              {isPdf ? (
                <iframe
                  src={fileUrl}
                  title="Document Preview"
                  style={{
                    width: "100%",
                    height: "640px",
                    border: "none",
                    borderRadius: "8px",
                    backgroundColor: "#ffffff",
                    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                  }}
                />
              ) : (
                <img
                  src={fileUrl}
                  alt="Document Preview"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "640px",
                    objectFit: "contain",
                    borderRadius: "8px",
                    boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                  }}
                />
              )}
            </div>
          </div>
        )}

        {/* Right Column: AI Document Assistant Chat Interface */}
        {activeTab === "chat" && (
          <div
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "12px",
              border: "1px solid #e2e8f0",
              boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
              display: "flex",
              flexDirection: "column",
              height: "min(720px, calc(100vh - 220px))",
              minHeight: "520px",
              overflow: "hidden",
            }}
          >
            {/* Chat Area Header */}
            <div
              style={{
                padding: "14px 20px",
                borderBottom: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                backgroundColor: "#f8fafc",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span
                  style={{
                    width: "32px",
                    height: "32px",
                    borderRadius: "8px",
                    backgroundColor: "#ede9fe",
                    color: "#6d28d9",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <SparklesIcon size={16} color="#6d28d9" />
                </span>
                <div>
                  <div style={{ fontWeight: "700", color: "#0f172a", fontSize: "14px" }}>
                    DocQA Inference Engine
                  </div>
                  <div style={{ fontSize: "11px", color: "#64748b" }}>
                    Session History: {currentChatHistory.length / 2} {currentChatHistory.length / 2 === 1 ? "turn" : "turns"}
                  </div>
                </div>
              </div>

              {/* Clear History Button */}
              {currentChatHistory.length > 0 && (
                <button
                  onClick={handleClearChat}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "5px",
                    padding: "6px 12px",
                    borderRadius: "6px",
                    backgroundColor: "#ffffff",
                    border: "1px solid #cbd5e1",
                    color: "#64748b",
                    fontSize: "12px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.12s ease",
                  }}
                  title="Clear conversation history for this document"
                >
                  <TrashIcon size={12} color="#64748b" />
                  Clear Chat
                </button>
              )}
            </div>

            {/* Chat Messages Feed */}
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                padding: "20px",
                display: "flex",
                flexDirection: "column",
                gap: "18px",
                backgroundColor: "#fcfdfe",
              }}
            >
              {currentChatHistory.length === 0 ? (
                /* Empty Chat: Welcome Card & Suggested Inquiry Prompts */
                <div style={{ padding: "16px 8px" }}>
                  <div
                    style={{
                      backgroundColor: "#ffffff",
                      borderRadius: "14px",
                      border: "1px solid #e2e8f0",
                      padding: "24px",
                      boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
                      marginBottom: "20px",
                      textAlign: "center",
                    }}
                  >
                    <div
                      style={{
                        width: "44px",
                        height: "44px",
                        borderRadius: "12px",
                        backgroundColor: "#ede9fe",
                        color: "#6d28d9",
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        marginBottom: "12px",
                      }}
                    >
                      <SparklesIcon size={22} color="#6d28d9" />
                    </div>
                    <h3 style={{ fontSize: "17px", fontWeight: "700", color: "#0f172a", margin: "0 0 6px" }}>
                      Ask Document QA Anything
                    </h3>
                    <p style={{ fontSize: "13px", color: "#64748b", margin: "0 auto", maxWidth: "520px", lineHeight: 1.5 }}>
                      Select a prompt below or type your custom query. The LayoutLM QA pipeline evaluates the document image and token positions to extract answers directly.
                    </p>
                  </div>

                  {/* Suggested Question Chips Grid */}
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: "700", color: "#475569", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "10px" }}>
                      Suggested Document Inquiries
                    </div>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                        gap: "10px",
                      }}
                    >
                      {SUGGESTED_QUESTIONS.map((sq, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleAsk(sq)}
                          disabled={asking}
                          style={{
                            textAlign: "left",
                            padding: "12px 14px",
                            borderRadius: "10px",
                            backgroundColor: "#ffffff",
                            border: "1px solid #e2e8f0",
                            color: "#1e293b",
                            fontSize: "13px",
                            fontWeight: "500",
                            cursor: asking ? "not-allowed" : "pointer",
                            transition: "all 0.15s ease",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: "8px",
                            boxShadow: "0 1px 2px rgba(15, 23, 42, 0.03)",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = "#c7d2fe";
                            e.currentTarget.style.backgroundColor = "#f5f3ff";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = "#e2e8f0";
                            e.currentTarget.style.backgroundColor = "#ffffff";
                          }}
                        >
                          <span>{sq}</span>
                          <span style={{ color: "#6d28d9", fontSize: "14px", fontWeight: "700" }}>&rarr;</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                /* Chat Messages List */
                currentChatHistory.map((item) => {
                  if (item.sender === "user") {
                    return (
                      <div
                        key={item.id}
                        style={{
                          display: "flex",
                          justifyContent: "flex-end",
                          alignItems: "flex-start",
                          gap: "10px",
                        }}
                      >
                        <div
                          style={{
                            maxWidth: "75%",
                            backgroundColor: "#0f172a",
                            color: "#ffffff",
                            padding: "12px 16px",
                            borderRadius: "14px 14px 2px 14px",
                            boxShadow: "0 2px 4px rgba(15, 23, 42, 0.08)",
                          }}
                        >
                          <div style={{ fontSize: "14px", lineHeight: 1.45, fontWeight: "500" }}>
                            {item.text}
                          </div>
                          <div style={{ fontSize: "10px", color: "#94a3b8", textAlign: "right", marginTop: "4px" }}>
                            {item.timestamp}
                          </div>
                        </div>

                        <span
                          style={{
                            width: "30px",
                            height: "30px",
                            borderRadius: "50%",
                            backgroundColor: "#f1f5f9",
                            color: "#475569",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            flexShrink: 0,
                          }}
                        >
                          <UserIcon size={14} color="#475569" />
                        </span>
                      </div>
                    );
                  }

                  // AI Response Message
                  return (
                    <div
                      key={item.id}
                      style={{
                        display: "flex",
                        justifyContent: "flex-start",
                        alignItems: "flex-start",
                        gap: "10px",
                      }}
                    >
                      <span
                        style={{
                          width: "32px",
                          height: "32px",
                          borderRadius: "50%",
                          backgroundColor: item.isError ? "#fee2e2" : "#ede9fe",
                          color: item.isError ? "#dc2626" : "#6d28d9",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexShrink: 0,
                          boxShadow: "0 1px 3px rgba(109, 40, 217, 0.15)",
                        }}
                      >
                        {item.isError ? (
                          <AlertTriangleIcon size={16} color="#dc2626" />
                        ) : (
                          <SparklesIcon size={16} color="#6d28d9" />
                        )}
                      </span>

                      <div
                        style={{
                          maxWidth: "85%",
                          backgroundColor: "#ffffff",
                          borderRadius: "14px 14px 14px 2px",
                          border: item.isError ? "1px solid #fecaca" : "1px solid #e2e8f0",
                          boxShadow: "0 2px 6px rgba(15, 23, 42, 0.04)",
                          overflow: "hidden",
                        }}
                      >
                        {/* Response Top Badge Strip */}
                        <div
                          style={{
                            padding: "8px 14px",
                            backgroundColor: item.isError ? "#fef2f2" : "#f8fafc",
                            borderBottom: item.isError ? "1px solid #fecaca" : "1px solid #f1f5f9",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            flexWrap: "wrap",
                            gap: "8px",
                            fontSize: "11px",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                            <span style={{ fontWeight: "700", color: item.isError ? "#dc2626" : "#6d28d9" }}>
                              {item.isError ? "Inference Notice" : "Grounded Extractive QA"}
                            </span>

                            {/* Source Type / Evidence Pill */}
                            {!item.isError && item.source_type && (
                              <span
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "4px",
                                  padding: "2px 8px",
                                  borderRadius: "6px",
                                  fontSize: "11px",
                                  fontWeight: "600",
                                  backgroundColor:
                                    item.source_type === "structured_field"
                                      ? "#ecfdf5"
                                      : item.source_type === "line_items"
                                      ? "#eef2ff"
                                      : item.source_type === "document_text"
                                      ? "#f5f3ff"
                                      : item.source_type === "unanswerable"
                                      ? "#f1f5f9"
                                      : "#fffbeb",
                                  color:
                                    item.source_type === "structured_field"
                                      ? "#065f46"
                                      : item.source_type === "line_items"
                                      ? "#4338ca"
                                      : item.source_type === "document_text"
                                      ? "#6d28d9"
                                      : item.source_type === "unanswerable"
                                      ? "#64748b"
                                      : "#b45309",
                                  border: `1px solid ${
                                    item.source_type === "structured_field"
                                      ? "#a7f3d0"
                                      : item.source_type === "line_items"
                                      ? "#c7d2fe"
                                      : item.source_type === "document_text"
                                      ? "#ddd6fe"
                                      : item.source_type === "unanswerable"
                                      ? "#cbd5e1"
                                      : "#fde68a"
                                  }`,
                                }}
                              >
                                {item.source_type === "structured_field"
                                  ? `Structured Field: ${item.field_name || "Canonical Field"}`
                                  : item.source_type === "line_items"
                                  ? "Line-Item Table Extraction"
                                  : item.source_type === "document_text"
                                  ? "Document Text Match"
                                  : item.source_type === "unanswerable"
                                  ? "Not Found in Document"
                                  : "LayoutLM Extractive QA"}
                              </span>
                            )}

                            {/* Multi-Page Source Attribution Indicator */}
                            {!item.isError && item.page_number != null && (
                              <span
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "4px",
                                  padding: "2px 8px",
                                  borderRadius: "6px",
                                  fontSize: "11px",
                                  fontWeight: "600",
                                  backgroundColor: "#ede9fe",
                                  color: "#5b21b6",
                                  border: "1px solid #ddd6fe",
                                }}
                                title={`Answer extracted from page ${item.page_number}${item.pages_searched ? ` of ${item.pages_searched} total searched pages` : ""}`}
                              >
                                <FileTextIcon size={12} color="#5b21b6" />
                                {item.pages_searched != null
                                  ? `Page ${item.page_number} of ${item.pages_searched}`
                                  : `Page ${item.page_number}`}
                              </span>
                            )}
                          </div>

                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            {item.confidence !== undefined && (
                              <ConfidenceBadge
                                score={item.confidence}
                                level={item.confidence_level}
                              />
                            )}
                            <span style={{ color: "#94a3b8" }}>{item.timestamp}</span>
                          </div>
                        </div>

                        {/* Main Response Body */}
                        <div style={{ padding: "14px 18px" }}>
                          {item.answer ? (
                            <div>
                              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px", marginBottom: "8px" }}>
                                <span style={{ fontSize: "12px", color: "#64748b", fontWeight: "600" }}>
                                  Extracted Answer:
                                </span>
                                <button
                                  onClick={() => handleCopyAnswer(item.answer, item.id)}
                                  style={{
                                    background: "none",
                                    border: "none",
                                    padding: "2px 6px",
                                    color: copiedAnswerId === item.id ? "#059669" : "#4f46e5",
                                    cursor: "pointer",
                                    fontSize: "11px",
                                    fontWeight: "600",
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: "4px",
                                  }}
                                  title="Copy answer text"
                                >
                                  {copiedAnswerId === item.id ? (
                                    <>
                                      <CheckIcon size={12} color="#059669" />
                                      Copied
                                    </>
                                  ) : (
                                    <>
                                      <CopyIcon size={12} color="#4f46e5" />
                                      Copy
                                    </>
                                  )}
                                </button>
                              </div>

                              <div
                                style={{
                                  fontSize: "16px",
                                  fontWeight: "600",
                                  color: "#0f172a",
                                  backgroundColor: "#f8fafc",
                                  padding: "12px 16px",
                                  borderRadius: "8px",
                                  border: "1px solid #e2e8f0",
                                  wordBreak: "break-word",
                                  whiteSpace: "pre-line",
                                  lineHeight: 1.6,
                                }}
                              >
                                {item.answer}
                              </div>
                            </div>
                          ) : (
                            <div
                              style={{
                                backgroundColor: "#f8fafc",
                                border: "1px solid #e2e8f0",
                                borderRadius: "8px",
                                padding: "12px 16px",
                                color: "#475569",
                                fontSize: "13px",
                                lineHeight: 1.5,
                                display: "flex",
                                alignItems: "center",
                                gap: "10px",
                              }}
                            >
                              <HelpCircleIcon size={18} color="#64748b" />
                              <span>{item.error || item.explanation || "The requested information could not be found or verified in this document."}</span>
                            </div>
                          )}

                          {/* Grounding Source Context Evidence */}
                          {item.source_text && (
                            <div
                              style={{
                                marginTop: "12px",
                                padding: "10px 12px",
                                backgroundColor: "#f8fafc",
                                borderRadius: "6px",
                                border: "1px dashed #cbd5e1",
                                fontSize: "12px",
                              }}
                            >
                              <div style={{ fontWeight: "700", color: "#475569", marginBottom: "4px", fontSize: "11px", textTransform: "uppercase" }}>
                                Grounding Document Evidence:
                              </div>
                              <div style={{ color: "#334155", fontStyle: "italic", lineHeight: 1.45 }}>
                                &ldquo;...{item.source_text}...&rdquo;
                              </div>
                            </div>
                          )}

                          {/* Model & Source Page Attribution Footer */}
                          <div
                            style={{
                              marginTop: "10px",
                              fontSize: "11px",
                              color: "#94a3b8",
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              flexWrap: "wrap",
                              gap: "8px",
                            }}
                          >
                            {item.model && (
                              <div>
                                Model: <span style={{ fontFamily: "monospace" }}>{item.model}</span>
                              </div>
                            )}

                            {!item.isError && item.page_number != null && (
                              <div style={{ color: "#64748b", fontWeight: "500" }}>
                                Source:{" "}
                                <strong style={{ color: "#334155" }}>
                                  Page {item.page_number}
                                </strong>
                                {item.pages_searched != null
                                  ? ` (${item.pages_searched} page${item.pages_searched === 1 ? "" : "s"} searched)`
                                  : ""}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}

              {/* In-Flight Asking Animation */}
              {asking && (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "flex-start",
                    alignItems: "flex-start",
                    gap: "10px",
                  }}
                >
                  <span
                    style={{
                      width: "32px",
                      height: "32px",
                      borderRadius: "50%",
                      backgroundColor: "#ede9fe",
                      color: "#6d28d9",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                    }}
                  >
                    <RefreshIcon size={16} spinning color="#6d28d9" />
                  </span>

                  <div
                    style={{
                      backgroundColor: "#ffffff",
                      borderRadius: "14px 14px 14px 2px",
                      border: "1px solid #e2e8f0",
                      padding: "14px 18px",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      boxShadow: "0 2px 4px rgba(15, 23, 42, 0.04)",
                    }}
                  >
                    <div style={{ display: "flex", gap: "4px" }}>
                      <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: "#6d28d9", animation: "bounce 1.4s infinite ease-in-out both" }} />
                      <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: "#6d28d9", animation: "bounce 1.4s infinite ease-in-out both 0.2s" }} />
                      <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: "#6d28d9", animation: "bounce 1.4s infinite ease-in-out both 0.4s" }} />
                    </div>
                    <span style={{ fontSize: "13px", color: "#64748b", fontWeight: "500" }}>
                      Analyzing document layout tokens & span logits...
                    </span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick Suggestions Strip when history exists */}
            {currentChatHistory.length > 0 && (
              <div
                style={{
                  padding: "8px 16px",
                  backgroundColor: "#f8fafc",
                  borderTop: "1px solid #e2e8f0",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  overflowX: "auto",
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ fontSize: "11px", color: "#94a3b8", fontWeight: "600" }}>Follow up:</span>
                {SUGGESTED_QUESTIONS.slice(0, 4).map((sq, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAsk(sq)}
                    disabled={asking}
                    style={{
                      padding: "4px 10px",
                      borderRadius: "12px",
                      backgroundColor: "#ffffff",
                      border: "1px solid #cbd5e1",
                      fontSize: "11px",
                      fontWeight: "500",
                      color: "#334155",
                      cursor: asking ? "not-allowed" : "pointer",
                      transition: "all 0.12s ease",
                    }}
                  >
                    {sq}
                  </button>
                ))}
              </div>
            )}

            {/* Chat Input Bar */}
            <div
              style={{
                padding: "16px 20px",
                borderTop: "1px solid #e2e8f0",
                backgroundColor: "#ffffff",
              }}
            >
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleAsk();
                }}
                style={{
                  display: "flex",
                  gap: "10px",
                  alignItems: "center",
                }}
              >
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Ask a question about this document (e.g. What is the total invoice amount?)..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={asking}
                  style={{
                    flex: 1,
                    padding: "12px 16px",
                    borderRadius: "10px",
                    border: "1px solid #cbd5e1",
                    fontSize: "14px",
                    color: "#0f172a",
                    backgroundColor: "#ffffff",
                    outline: "none",
                    transition: "border-color 0.15s ease",
                  }}
                />

                <button
                  type="submit"
                  disabled={asking || !question.trim()}
                  aria-label="Send Question"
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                    padding: "12px 20px",
                    borderRadius: "10px",
                    backgroundColor: asking || !question.trim() ? "#cbd5e1" : "#4f46e5",
                    color: "#ffffff",
                    border: "none",
                    fontWeight: "600",
                    fontSize: "14px",
                    cursor: asking || !question.trim() ? "not-allowed" : "pointer",
                    boxShadow: asking || !question.trim() ? "none" : "0 1px 3px rgba(79, 70, 229, 0.3)",
                    transition: "background-color 0.15s ease",
                  }}
                >
                  <SendIcon size={15} color="#ffffff" />
                  <span>{asking ? "Analyzing..." : "Ask"}</span>
                </button>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Subcomponents ────────────────────────────────────────────────────────────

function ContextFieldRow({ label, value, highlight = false, isMono = false, textColor }) {
  const displayVal = value && value !== "Not found" ? value : "—";
  const isAvailable = displayVal !== "—";

  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "8px" }}>
      <span style={{ color: "#64748b", fontWeight: "500", whiteSpace: "nowrap" }}>{label}:</span>
      <span
        style={{
          fontWeight: highlight ? "700" : isAvailable ? "600" : "400",
          color: textColor || (isAvailable ? (highlight ? "#0f172a" : "#334155") : "#94a3b8"),
          fontFamily: isMono ? "monospace" : "inherit",
          textAlign: "right",
          wordBreak: "break-all",
        }}
      >
        {displayVal}
      </span>
    </div>
  );
}

function ConfidenceBadge({ score, level }) {
  const percentage = (score * 100).toFixed(1);
  const isHigh = score >= 0.8 || level === "VERY_HIGH" || level === "HIGH";
  const isMedium = (score >= 0.5 && score < 0.8) || level === "MEDIUM";

  let bg = "#fee2e2";
  let text = "#991b1b";
  let border = "#fecaca";

  if (isHigh) {
    bg = "#ecfdf5";
    text = "#065f46";
    border = "#a7f3d0";
  } else if (isMedium) {
    bg = "#fffbeb";
    text = "#92400e";
    border = "#fde68a";
  }

  return (
    <span
      style={{
        padding: "2px 8px",
        borderRadius: "10px",
        fontWeight: "600",
        fontSize: "11px",
        backgroundColor: bg,
        color: text,
        border: `1px solid ${border}`,
      }}
      title={`Confidence score: ${score}`}
    >
      {percentage}% ({level || (isHigh ? "HIGH" : isMedium ? "MED" : "LOW")})
    </span>
  );
}
