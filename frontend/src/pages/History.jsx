import { useEffect, useState, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import { getDocuments, deleteDocument, getStats, getDocumentFileUrl } from "../services/api";

// ── SVG Icon Helpers ─────────────────────────────────────────────────────────

function SearchIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function RefreshIcon({ size = 15, spinning = false, color = "currentColor" }) {
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

function FilterIcon({ size = 15, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
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

function CheckCircleIcon({ size = 15, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
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

function EyeIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function DownloadIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
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

function ArrowRightIcon({ size = 13, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

function CloseIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function ExternalLinkIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
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

function LightningIcon({ size = 14, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function History() {
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [successBanner, setSuccessBanner] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  // Search, Filters & Sorting
  const [searchQuery, setSearchQuery] = useState("");
  const [methodFilter, setMethodFilter] = useState("all"); // 'all' | 'layoutlm' | 'baseline'
  const [statusFilter, setStatusFilter] = useState("all"); // 'all' | 'processed' | 'failed'
  const [verificationFilter, setVerificationFilter] = useState("all"); // 'all' | 'verified' | 'pending'
  const [sortBy, setSortBy] = useState("date_desc"); // 'date_desc' | 'date_asc' | 'amount_desc' | 'amount_asc' | 'name_asc' | 'name_desc'

  // Modals & Drawers
  const [selectedDoc, setSelectedDoc] = useState(null); // Document details modal
  const [deleteTarget, setDeleteTarget] = useState(null); // { id, filename }
  const [isDeleting, setIsDeleting] = useState(false);
  const [copiedKey, setCopiedKey] = useState(null);

  // Load documents and stats from backend
  const loadData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    setError("");

    try {
      const [statsData, docsData] = await Promise.all([
        getStats().catch(() => null),
        getDocuments({ limit: 100 }).catch(() => ({ documents: [] })),
      ]);

      setStats(statsData);
      setDocuments(docsData?.documents || []);
      setLastUpdated(new Date());
    } catch {
      setError("Failed to load document records from backend. Please verify the server is running.");
    } finally {
      setLoading(false);
      if (isRefresh) setRefreshing(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    let active = true;

    async function fetchInitial() {
      try {
        const [statsData, docsData] = await Promise.all([
          getStats().catch(() => null),
          getDocuments({ limit: 100 }).catch(() => ({ documents: [] })),
        ]);

        if (active) {
          setStats(statsData);
          setDocuments(docsData?.documents || []);
          setLastUpdated(new Date());
        }
      } catch {
        if (active) {
          setError("Failed to load document records from backend. Please verify the server is running.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    fetchInitial();

    return () => {
      active = false;
    };
  }, []);

  // Handle document deletion
  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;

    setIsDeleting(true);
    try {
      await deleteDocument(deleteTarget.id);
      setSuccessBanner(`Document "${deleteTarget.filename}" deleted successfully.`);
      setTimeout(() => setSuccessBanner(""), 4500);

      // Close details modal if open with this doc
      if (selectedDoc && selectedDoc.id === deleteTarget.id) {
        setSelectedDoc(null);
      }

      setDeleteTarget(null);
      // Reload updated records and stats
      await loadData(false);
    } catch (err) {
      alert(`Deletion failed: ${err.message}`);
    } finally {
      setIsDeleting(false);
    }
  };

  // Export JSON file
  const handleExportJSON = (doc) => {
    const data = JSON.stringify(doc, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = doc.filename ? `${doc.filename}.json` : "document-metadata.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  // Copy to clipboard helper
  const handleCopy = (text, key) => {
    if (!navigator.clipboard || !text) return;
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1800);
  };

  // Filter and sort computation
  const filteredDocuments = useMemo(() => {
    let result = documents.filter((doc) => {
      const isLayoutLM =
        doc.extraction_method === "layoutlm" ||
        Boolean(doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0);

      const activeFields =
        isLayoutLM && doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0
          ? doc.layoutlm_fields
          : (doc.invoice_fields || doc.extracted_fields || {});

      // Method filter
      if (methodFilter === "layoutlm" && !isLayoutLM) return false;
      if (methodFilter === "baseline" && isLayoutLM) return false;

      // Status filter
      if (statusFilter === "processed" && doc.processing_status !== "processed") return false;
      if (statusFilter === "failed" && doc.processing_status !== "failed") return false;

      // Verification filter
      if (verificationFilter === "verified" && !doc.verified) return false;
      if (verificationFilter === "pending" && doc.verified) return false;

      // Search query filter (matches filename, vendor, customer, invoice number, GSTIN, ID)
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const filename = (doc.filename || "").toLowerCase();
        const docType = (doc.document_type || "").toLowerCase();
        const vendor = (activeFields.vendor_name || activeFields.company_name || "").toLowerCase();
        const customer = (activeFields.customer_name || "").toLowerCase();
        const invNum = (activeFields.invoice_number || "").toLowerCase();
        const gstin = (
          activeFields.vendor_gstin ||
          activeFields.customer_gstin ||
          activeFields.gst_number ||
          ""
        ).toLowerCase();
        const docId = (doc.id || "").toLowerCase();

        return (
          filename.includes(q) ||
          docType.includes(q) ||
          vendor.includes(q) ||
          customer.includes(q) ||
          invNum.includes(q) ||
          gstin.includes(q) ||
          docId.includes(q)
        );
      }

      return true;
    });

    // Sorting
    result = [...result].sort((a, b) => {
      if (sortBy === "date_desc") {
        return new Date(b.uploaded_at || 0) - new Date(a.uploaded_at || 0);
      }
      if (sortBy === "date_asc") {
        return new Date(a.uploaded_at || 0) - new Date(b.uploaded_at || 0);
      }
      if (sortBy === "amount_desc" || sortBy === "amount_asc") {
        const getAmt = (d) => {
          const isLLM =
            d.extraction_method === "layoutlm" ||
            Boolean(d.layoutlm_fields && Object.keys(d.layoutlm_fields).length > 0);
          const f = isLLM && d.layoutlm_fields ? d.layoutlm_fields : (d.invoice_fields || d.extracted_fields || {});
          const val = f.total_amount || f.subtotal || 0;
          const num = typeof val === "string" ? parseFloat(val.replace(/[^0-9.-]/g, "")) : Number(val);
          return isNaN(num) ? 0 : num;
        };
        const amtA = getAmt(a);
        const amtB = getAmt(b);
        return sortBy === "amount_desc" ? amtB - amtA : amtA - amtB;
      }
      if (sortBy === "name_asc") {
        return (a.filename || "").localeCompare(b.filename || "");
      }
      if (sortBy === "name_desc") {
        return (b.filename || "").localeCompare(a.filename || "");
      }
      return 0;
    });

    return result;
  }, [documents, searchQuery, methodFilter, statusFilter, verificationFilter, sortBy]);

  // Check if any filter is active
  const hasActiveFilters =
    searchQuery.trim() !== "" ||
    methodFilter !== "all" ||
    statusFilter !== "all" ||
    verificationFilter !== "all" ||
    sortBy !== "date_desc";

  const clearAllFilters = () => {
    setSearchQuery("");
    setMethodFilter("all");
    setStatusFilter("all");
    setVerificationFilter("all");
    setSortBy("date_desc");
  };

  // Metrics from real backend data
  const totalBackendDocs = stats?.total_documents ?? documents.length;
  const verifiedCount = stats?.verified ?? documents.filter((d) => d.verified).length;
  const pendingCount = stats?.unverified ?? documents.filter((d) => !d.verified).length;
  const layoutlmCount = documents.filter(
    (d) =>
      d.extraction_method === "layoutlm" ||
      Boolean(d.layoutlm_fields && Object.keys(d.layoutlm_fields).length > 0)
  ).length;

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
          marginBottom: "24px",
          paddingBottom: "22px",
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
              Document History & Repository
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
                backgroundColor: "#eef2ff",
                color: "#4f46e5",
                border: "1px solid #c7d2fe",
              }}
            >
              <FileTextIcon size={12} color="#4f46e5" />
              {totalBackendDocs} {totalBackendDocs === 1 ? "Record" : "Records"}
            </span>
          </div>

          <p
            style={{
              color: "#475569",
              fontSize: "14px",
              margin: 0,
              maxWidth: "720px",
              lineHeight: 1.5,
            }}
          >
            Comprehensive repository of all ingested enterprise documents. Audit extraction methods, filter by verification status, review structured fields, and export audit trails.
          </p>

          {lastUpdated && (
            <div style={{ fontSize: "12px", color: "#94a3b8", marginTop: "6px" }}>
              Last synchronized: {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>

        {/* Action Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
          <button
            onClick={() => loadData(true)}
            disabled={refreshing || loading}
            aria-label="Refresh Documents"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "9px 16px",
              backgroundColor: "#ffffff",
              color: "#334155",
              border: "1px solid #cbd5e1",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "13px",
              cursor: refreshing || loading ? "not-allowed" : "pointer",
              transition: "all 0.15s ease",
              boxShadow: "0 1px 2px rgba(15, 23, 42, 0.05)",
            }}
          >
            <RefreshIcon size={14} spinning={refreshing || loading} color="#475569" />
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>

          <Link
            to="/upload"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "9px 18px",
              backgroundColor: "#4f46e5",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "13px",
              boxShadow: "0 1px 3px rgba(79, 70, 229, 0.25)",
              transition: "background-color 0.15s ease",
            }}
          >
            <span>+</span> Upload Documents
          </Link>
        </div>
      </div>

      {/* ── Success and Error Banners ───────────────────────────────────────── */}
      {successBanner && (
        <div
          style={{
            backgroundColor: "#ecfdf5",
            border: "1px solid #a7f3d0",
            color: "#065f46",
            padding: "12px 18px",
            borderRadius: "8px",
            marginBottom: "20px",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <CheckCircleIcon size={16} color="#059669" />
          <span>{successBanner}</span>
        </div>
      )}

      {error && (
        <div
          style={{
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            color: "#b91c1c",
            padding: "14px 18px",
            borderRadius: "8px",
            marginBottom: "20px",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "12px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <AlertTriangleIcon size={16} color="#b91c1c" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => loadData(true)}
            style={{
              background: "none",
              border: "none",
              color: "#b91c1c",
              textDecoration: "underline",
              cursor: "pointer",
              fontSize: "12px",
              fontWeight: "600",
            }}
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* ── 2. Top Metric Cards (Quick Filter Cards) ────────────────────────── */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "14px",
          marginBottom: "24px",
        }}
      >
        <MetricCard
          title="Total Documents"
          value={totalBackendDocs}
          subtitle="All ingested files in database"
          active={methodFilter === "all" && verificationFilter === "all" && statusFilter === "all"}
          badge="Live DB"
          badgeColor="#eef2ff"
          badgeTextColor="#4f46e5"
          onClick={clearAllFilters}
        />

        <MetricCard
          title="Verified Documents"
          value={verifiedCount}
          subtitle="Audited and approved by human"
          active={verificationFilter === "verified"}
          badge="Verified"
          badgeColor="#ecfdf5"
          badgeTextColor="#059669"
          onClick={() => {
            setVerificationFilter(verificationFilter === "verified" ? "all" : "verified");
          }}
        />

        <MetricCard
          title="Pending Verification"
          value={pendingCount}
          subtitle="Requires human-in-the-loop review"
          active={verificationFilter === "pending"}
          badge="Action Needed"
          badgeColor="#fff7ed"
          badgeTextColor="#c2410c"
          onClick={() => {
            setVerificationFilter(verificationFilter === "pending" ? "all" : "pending");
          }}
        />

        <MetricCard
          title="LayoutLMv3 Powered"
          value={layoutlmCount}
          subtitle="Spatial transformer extracted"
          active={methodFilter === "layoutlm"}
          badge="Multimodal"
          badgeColor="#f5f3ff"
          badgeTextColor="#6d28d9"
          onClick={() => {
            setMethodFilter(methodFilter === "layoutlm" ? "all" : "layoutlm");
          }}
        />
      </div>

      {/* ── 3. Search and Filtering Controls ─────────────────────────────────── */}
      <div
        style={{
          backgroundColor: "#ffffff",
          padding: "18px 20px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
          marginBottom: "20px",
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "12px",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Search Box */}
          <div
            style={{
              position: "relative",
              flex: "1 1 320px",
              minWidth: "260px",
            }}
          >
            <div
              style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
                color: "#94a3b8",
                pointerEvents: "none",
                display: "flex",
                alignItems: "center",
              }}
            >
              <SearchIcon size={16} />
            </div>
            <input
              type="text"
              placeholder="Search by filename, vendor, customer, invoice #, or GSTIN..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: "100%",
                padding: "9px 36px 9px 36px",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
                fontSize: "13px",
                color: "#0f172a",
                backgroundColor: "#ffffff",
                outline: "none",
                boxSizing: "border-box",
                transition: "border-color 0.15s ease",
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                aria-label="Clear Search"
                style={{
                  position: "absolute",
                  right: "10px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "none",
                  border: "none",
                  color: "#94a3b8",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  padding: "4px",
                }}
              >
                <CloseIcon size={14} />
              </button>
            )}
          </div>

          {/* Filter Dropdowns */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" }}>
            {/* Extraction Method Filter */}
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <select
                value={methodFilter}
                onChange={(e) => setMethodFilter(e.target.value)}
                aria-label="Filter by extraction method"
                style={{
                  padding: "8px 12px",
                  fontSize: "13px",
                  border: "1px solid #cbd5e1",
                  borderRadius: "8px",
                  backgroundColor: methodFilter !== "all" ? "#f5f3ff" : "#ffffff",
                  color: methodFilter !== "all" ? "#6d28d9" : "#334155",
                  fontWeight: methodFilter !== "all" ? "600" : "500",
                  outline: "none",
                  cursor: "pointer",
                }}
              >
                <option value="all">All Methods</option>
                <option value="layoutlm">⚡ LayoutLMv3</option>
                <option value="baseline">📄 Baseline OCR</option>
              </select>
            </div>

            {/* Processing Status Filter */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              aria-label="Filter by processing status"
              style={{
                padding: "8px 12px",
                fontSize: "13px",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
                backgroundColor: statusFilter !== "all" ? "#ecfdf5" : "#ffffff",
                color: statusFilter !== "all" ? "#065f46" : "#334155",
                fontWeight: statusFilter !== "all" ? "600" : "500",
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="all">All Statuses</option>
              <option value="processed">Processed</option>
              <option value="failed">Failed</option>
            </select>

            {/* Verification Status Filter */}
            <select
              value={verificationFilter}
              onChange={(e) => setVerificationFilter(e.target.value)}
              aria-label="Filter by verification status"
              style={{
                padding: "8px 12px",
                fontSize: "13px",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
                backgroundColor: verificationFilter !== "all" ? "#fff7ed" : "#ffffff",
                color: verificationFilter !== "all" ? "#c2410c" : "#334155",
                fontWeight: verificationFilter !== "all" ? "600" : "500",
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="all">All Verification</option>
              <option value="verified">✓ Verified Only</option>
              <option value="pending">⚠ Pending Review</option>
            </select>

            {/* Sort Order */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              aria-label="Sort documents"
              style={{
                padding: "8px 12px",
                fontSize: "13px",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
                backgroundColor: "#ffffff",
                color: "#334155",
                fontWeight: "500",
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="date_desc">Upload Date (Newest first)</option>
              <option value="date_asc">Upload Date (Oldest first)</option>
              <option value="amount_desc">Total Amount (High to Low)</option>
              <option value="amount_asc">Total Amount (Low to High)</option>
              <option value="name_asc">Filename (A → Z)</option>
              <option value="name_desc">Filename (Z → A)</option>
            </select>

            {/* Clear Filters Action */}
            {hasActiveFilters && (
              <button
                onClick={clearAllFilters}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "8px 12px",
                  fontSize: "12px",
                  fontWeight: "600",
                  color: "#dc2626",
                  backgroundColor: "#fee2e2",
                  border: "1px solid #fecaca",
                  borderRadius: "8px",
                  cursor: "pointer",
                  transition: "background-color 0.15s ease",
                }}
              >
                <CloseIcon size={12} color="#dc2626" />
                Clear Filters
              </button>
            )}
          </div>
        </div>

        {/* Active Filter Chips Bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "8px",
            marginTop: "14px",
            paddingTop: "12px",
            borderTop: "1px solid #f1f5f9",
            fontSize: "12px",
            color: "#64748b",
          }}
        >
          <div>
            Showing <strong>{filteredDocuments.length}</strong> of <strong>{documents.length}</strong> loaded documents
            {hasActiveFilters && (
              <span style={{ marginLeft: "8px", color: "#4f46e5", fontWeight: "600" }}>
                (Filters active)
              </span>
            )}
          </div>

          <div style={{ display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ color: "#94a3b8", fontSize: "11px" }}>Quick slices:</span>
            <FilterChip
              label="All"
              active={!hasActiveFilters}
              onClick={clearAllFilters}
            />
            <FilterChip
              label="Needs Review"
              active={verificationFilter === "pending"}
              onClick={() => setVerificationFilter(verificationFilter === "pending" ? "all" : "pending")}
            />
            <FilterChip
              label="Verified"
              active={verificationFilter === "verified"}
              onClick={() => setVerificationFilter(verificationFilter === "verified" ? "all" : "verified")}
            />
            <FilterChip
              label="LayoutLMv3"
              active={methodFilter === "layoutlm"}
              onClick={() => setMethodFilter(methodFilter === "layoutlm" ? "all" : "layoutlm")}
            />
          </div>
        </div>
      </div>

      {/* ── 4. Document History Table ────────────────────────────────────────── */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
          overflow: "hidden",
        }}
      >
        {loading ? (
          <TableSkeleton />
        ) : filteredDocuments.length === 0 ? (
          <EmptyState
            hasFilters={hasActiveFilters}
            totalDocs={documents.length}
            onClearFilters={clearAllFilters}
          />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "13px",
                textAlign: "left",
              }}
            >
              <thead>
                <tr
                  style={{
                    backgroundColor: "#f8fafc",
                    borderBottom: "1px solid #e2e8f0",
                    color: "#64748b",
                    fontSize: "11px",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  <th style={{ padding: "14px 18px", fontWeight: "600" }}>Document Filename</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Type</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Uploaded At</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Method</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Processing</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Verification</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Vendor Name</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Invoice #</th>
                  <th style={{ padding: "14px 14px", fontWeight: "600" }}>Total Amount</th>
                  <th style={{ padding: "14px 18px", fontWeight: "600", textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredDocuments.map((doc) => {
                  const isLayoutLM =
                    doc.extraction_method === "layoutlm" ||
                    Boolean(doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0);

                  const activeFields =
                    isLayoutLM && doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0
                      ? doc.layoutlm_fields
                      : (doc.invoice_fields || doc.extracted_fields || {});

                  const vendorName = activeFields.vendor_name || activeFields.company_name || null;
                  const invoiceNumber = activeFields.invoice_number || null;
                  const rawAmount = activeFields.total_amount || activeFields.subtotal || null;
                  const currency =
                    activeFields.currency && activeFields.currency !== "Not found"
                      ? activeFields.currency
                      : rawAmount
                      ? "₹"
                      : "";

                  const uploadDate = doc.uploaded_at ? new Date(doc.uploaded_at) : null;
                  const formattedDate =
                    uploadDate && !isNaN(uploadDate.getTime())
                      ? uploadDate.toLocaleDateString(undefined, {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })
                      : "—";
                  const formattedTime =
                    uploadDate && !isNaN(uploadDate.getTime())
                      ? uploadDate.toLocaleTimeString(undefined, {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : "";

                  const coverageScore = doc.coverage_score ?? 0;
                  const modelConfidence = doc.model_confidence ?? doc.confidence_score ?? 0;

                  return (
                    <tr
                      key={doc.id}
                      style={{
                        borderBottom: "1px solid #e2e8f0",
                        transition: "background-color 0.12s ease",
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f8fafc")}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                    >
                      {/* Document Filename */}
                      <td style={{ padding: "14px 18px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <span
                            style={{
                              width: "34px",
                              height: "34px",
                              borderRadius: "8px",
                              backgroundColor: isLayoutLM ? "#f5f3ff" : "#f1f5f9",
                              color: isLayoutLM ? "#6d28d9" : "#475569",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              flexShrink: 0,
                              border: isLayoutLM ? "1px solid #ddd6fe" : "1px solid #e2e8f0",
                            }}
                          >
                            <FileTextIcon size={16} />
                          </span>
                          <div>
                            <button
                              onClick={() => setSelectedDoc(doc)}
                              style={{
                                background: "none",
                                border: "none",
                                padding: 0,
                                fontWeight: "600",
                                color: "#0f172a",
                                fontSize: "13px",
                                textAlign: "left",
                                cursor: "pointer",
                                textDecoration: "none",
                                display: "block",
                              }}
                              onMouseEnter={(e) => (e.currentTarget.style.color = "#4f46e5")}
                              onMouseLeave={(e) => (e.currentTarget.style.color = "#0f172a")}
                              title="Click to view full document details"
                            >
                              {doc.filename}
                            </button>
                            <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "2px" }}>
                              <span style={{ fontSize: "11px", color: "#94a3b8" }}>
                                {doc.page_count ? `${doc.page_count} page${doc.page_count > 1 ? "s" : ""}` : "1 page"}
                              </span>
                              <span style={{ fontSize: "11px", color: "#cbd5e1" }}>•</span>
                              <span style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "monospace" }}>
                                {doc.id ? doc.id.slice(0, 8) : ""}
                              </span>
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Document Type */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        <span
                          style={{
                            padding: "3px 8px",
                            borderRadius: "6px",
                            backgroundColor: "#f1f5f9",
                            color: "#334155",
                            fontSize: "12px",
                            fontWeight: "500",
                          }}
                        >
                          {doc.document_type ? doc.document_type.charAt(0).toUpperCase() + doc.document_type.slice(1) : "Invoice"}
                        </span>
                      </td>

                      {/* Upload Date & Time */}
                      <td style={{ padding: "14px 14px", color: "#334155", whiteSpace: "nowrap" }}>
                        <div style={{ fontWeight: "500" }}>{formattedDate}</div>
                        {formattedTime && (
                          <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "1px" }}>
                            {formattedTime}
                          </div>
                        )}
                      </td>

                      {/* Extraction Method */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {isLayoutLM ? (
                          <span
                            style={{
                              padding: "4px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#f5f3ff",
                              color: "#6d28d9",
                              border: "1px solid #ddd6fe",
                              fontWeight: "600",
                              fontSize: "11px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "4px",
                            }}
                            title={`LayoutLMv3 Spatial Transformer: ${modelConfidence}% confidence`}
                          >
                            <LightningIcon size={12} color="#6d28d9" />
                            LayoutLMv3
                          </span>
                        ) : (
                          <span
                            style={{
                              padding: "4px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#f1f5f9",
                              color: "#475569",
                              border: "1px solid #e2e8f0",
                              fontWeight: "600",
                              fontSize: "11px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "4px",
                            }}
                            title={`Baseline OCR Heuristics: ${coverageScore}% field coverage`}
                          >
                            <FileTextIcon size={12} color="#475569" />
                            Baseline OCR
                          </span>
                        )}
                      </td>

                      {/* Processing Status */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {doc.processing_status === "processed" ? (
                          <span
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "5px",
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#ecfdf5",
                              color: "#059669",
                              fontWeight: "600",
                              fontSize: "11px",
                              border: "1px solid #a7f3d0",
                            }}
                          >
                            <span
                              style={{
                                width: "6px",
                                height: "6px",
                                borderRadius: "50%",
                                backgroundColor: "#059669",
                              }}
                            />
                            Processed
                          </span>
                        ) : (
                          <span
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "5px",
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#fef2f2",
                              color: "#dc2626",
                              fontWeight: "600",
                              fontSize: "11px",
                              border: "1px solid #fecaca",
                            }}
                          >
                            <span
                              style={{
                                width: "6px",
                                height: "6px",
                                borderRadius: "50%",
                                backgroundColor: "#dc2626",
                              }}
                            />
                            Failed
                          </span>
                        )}
                      </td>

                      {/* Verification Status */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {doc.verified ? (
                          <span
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "5px",
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#ecfdf5",
                              color: "#15803d",
                              fontWeight: "600",
                              fontSize: "11px",
                              border: "1px solid #bbf7d0",
                            }}
                          >
                            <CheckCircleIcon size={12} color="#15803d" />
                            Verified
                          </span>
                        ) : (
                          <span
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "5px",
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#fff7ed",
                              color: "#c2410c",
                              fontWeight: "600",
                              fontSize: "11px",
                              border: "1px solid #fed7aa",
                            }}
                          >
                            <AlertTriangleIcon size={12} color="#c2410c" />
                            Needs Review
                          </span>
                        )}
                      </td>

                      {/* Vendor Name */}
                      <td style={{ padding: "14px 14px", maxWidth: "200px" }}>
                        <div
                          style={{
                            fontWeight: vendorName && vendorName !== "Not found" ? "600" : "400",
                            color: vendorName && vendorName !== "Not found" ? "#0f172a" : "#94a3b8",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                          title={vendorName || "Not extracted"}
                        >
                          {vendorName && vendorName !== "Not found" ? vendorName : "—"}
                        </div>
                      </td>

                      {/* Invoice Number */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {invoiceNumber && invoiceNumber !== "Not found" ? (
                          <span
                            style={{
                              fontFamily: "monospace",
                              fontSize: "12px",
                              fontWeight: "600",
                              color: "#334155",
                              backgroundColor: "#f1f5f9",
                              padding: "2px 6px",
                              borderRadius: "4px",
                            }}
                          >
                            {invoiceNumber}
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>—</span>
                        )}
                      </td>

                      {/* Total Amount */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {rawAmount && rawAmount !== "Not found" ? (
                          <span style={{ fontWeight: "700", color: "#0f172a" }}>
                            {currency ? `${currency} ` : ""}{rawAmount}
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>—</span>
                        )}
                      </td>

                      {/* Actions */}
                      <td style={{ padding: "14px 18px", textAlign: "right", whiteSpace: "nowrap" }}>
                        <div style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                          {/* Review & Verify CTA */}
                          <Link
                            to={`/review/${doc.id}`}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "4px",
                              padding: "6px 12px",
                              backgroundColor: doc.verified ? "#f1f5f9" : "#4f46e5",
                              color: doc.verified ? "#334155" : "#ffffff",
                              border: doc.verified ? "1px solid #cbd5e1" : "none",
                              borderRadius: "6px",
                              fontSize: "12px",
                              fontWeight: "600",
                              textDecoration: "none",
                              transition: "all 0.15s ease",
                            }}
                            title={doc.verified ? "View verification review" : "Continue human verification"}
                          >
                            <span>{doc.verified ? "Review" : "Verify"}</span>
                            <ArrowRightIcon size={11} color={doc.verified ? "#475569" : "#ffffff"} />
                          </Link>

                          {/* Quick Inspect Details */}
                          <button
                            onClick={() => setSelectedDoc(doc)}
                            aria-label={`View details for ${doc.filename}`}
                            style={{
                              padding: "6px 9px",
                              backgroundColor: "#ffffff",
                              color: "#475569",
                              border: "1px solid #cbd5e1",
                              borderRadius: "6px",
                              fontSize: "12px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                            title="Inspect structured document fields"
                          >
                            <EyeIcon size={14} color="#475569" />
                          </button>

                          {/* Export JSON */}
                          <button
                            onClick={() => handleExportJSON(doc)}
                            aria-label={`Export JSON for ${doc.filename}`}
                            style={{
                              padding: "6px 9px",
                              backgroundColor: "#ffffff",
                              color: "#475569",
                              border: "1px solid #cbd5e1",
                              borderRadius: "6px",
                              fontSize: "12px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                            title="Export document record as JSON"
                          >
                            <DownloadIcon size={14} color="#475569" />
                          </button>

                          {/* Delete Document */}
                          <button
                            onClick={() => setDeleteTarget({ id: doc.id, filename: doc.filename })}
                            aria-label={`Delete ${doc.filename}`}
                            style={{
                              padding: "6px 9px",
                              backgroundColor: "#ffffff",
                              color: "#dc2626",
                              border: "1px solid #fecaca",
                              borderRadius: "6px",
                              fontSize: "12px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                            title="Delete document and stored file"
                          >
                            <TrashIcon size={14} color="#dc2626" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── 5. Document Details Modal ────────────────────────────────────────── */}
      {selectedDoc && (
        <DocumentDetailsModal
          doc={selectedDoc}
          onClose={() => setSelectedDoc(null)}
          onExport={() => handleExportJSON(selectedDoc)}
          onDelete={() => {
            setDeleteTarget({ id: selectedDoc.id, filename: selectedDoc.filename });
          }}
          copiedKey={copiedKey}
          onCopy={handleCopy}
        />
      )}

      {/* ── 6. Delete Confirmation Dialog ───────────────────────────────────── */}
      {deleteTarget && (
        <DeleteConfirmationModal
          filename={deleteTarget.filename}
          isDeleting={isDeleting}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}

// ── Subcomponents ────────────────────────────────────────────────────────────

function MetricCard({ title, value, subtitle, active, badge, badgeColor, badgeTextColor, onClick }) {
  return (
    <div
      onClick={onClick}
      style={{
        backgroundColor: "#ffffff",
        padding: "18px 20px",
        borderRadius: "12px",
        border: active ? "2px solid #4f46e5" : "1px solid #e2e8f0",
        boxShadow: active
          ? "0 4px 12px rgba(79, 70, 229, 0.12)"
          : "0 1px 3px rgba(15, 23, 42, 0.04)",
        cursor: "pointer",
        transition: "all 0.15s ease",
        position: "relative",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
        <span style={{ fontSize: "12px", color: "#64748b", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.04em" }}>
          {title}
        </span>
        {badge && (
          <span
            style={{
              padding: "2px 8px",
              fontSize: "11px",
              fontWeight: "600",
              borderRadius: "10px",
              backgroundColor: badgeColor,
              color: badgeTextColor,
            }}
          >
            {badge}
          </span>
        )}
      </div>

      <div style={{ fontSize: "28px", fontWeight: "700", color: "#0f172a", marginBottom: "4px" }}>
        {value}
      </div>

      <div style={{ fontSize: "12px", color: "#94a3b8" }}>
        {subtitle}
      </div>
    </div>
  );
}

function FilterChip({ label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "3px 9px",
        borderRadius: "12px",
        fontSize: "11px",
        fontWeight: "600",
        border: active ? "1px solid #4f46e5" : "1px solid #e2e8f0",
        backgroundColor: active ? "#eef2ff" : "#f8fafc",
        color: active ? "#4f46e5" : "#475569",
        cursor: "pointer",
        transition: "all 0.12s ease",
      }}
    >
      {label}
    </button>
  );
}

function TableSkeleton() {
  return (
    <div style={{ padding: "20px" }}>
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "16px 0",
            borderBottom: i < 5 ? "1px solid #f1f5f9" : "none",
            gap: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px", flex: 2 }}>
            <div style={{ width: "34px", height: "34px", borderRadius: "8px", backgroundColor: "#f1f5f9" }} />
            <div style={{ width: "160px", height: "14px", borderRadius: "4px", backgroundColor: "#f1f5f9" }} />
          </div>
          <div style={{ width: "70px", height: "14px", borderRadius: "4px", backgroundColor: "#f1f5f9", flex: 1 }} />
          <div style={{ width: "100px", height: "14px", borderRadius: "4px", backgroundColor: "#f1f5f9", flex: 1 }} />
          <div style={{ width: "90px", height: "14px", borderRadius: "4px", backgroundColor: "#f1f5f9", flex: 1 }} />
          <div style={{ width: "120px", height: "14px", borderRadius: "4px", backgroundColor: "#f1f5f9", flex: 1 }} />
        </div>
      ))}
    </div>
  );
}

function EmptyState({ hasFilters, totalDocs, onClearFilters }) {
  return (
    <div style={{ padding: "60px 20px", textAlign: "center" }}>
      <div
        style={{
          width: "48px",
          height: "48px",
          borderRadius: "12px",
          backgroundColor: "#f1f5f9",
          color: "#94a3b8",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "14px",
        }}
      >
        <FilterIcon size={24} />
      </div>

      <h3 style={{ fontSize: "16px", fontWeight: "600", color: "#0f172a", margin: "0 0 6px" }}>
        {hasFilters ? "No matching documents found" : "Repository is empty"}
      </h3>

      <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 18px", maxWidth: "420px", marginInline: "auto" }}>
        {hasFilters
          ? "No documents match your current search terms or filter selections. Try clearing your filters or searching with different keywords."
          : "You haven't uploaded any documents yet. Ingest your first PDF or image to start automated extraction."}
      </p>

      {hasFilters ? (
        <button
          onClick={onClearFilters}
          style={{
            padding: "8px 18px",
            backgroundColor: "#4f46e5",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            fontSize: "13px",
            fontWeight: "600",
            cursor: "pointer",
          }}
        >
          Reset All Filters
        </button>
      ) : (
        <Link
          to="/upload"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            padding: "8px 18px",
            backgroundColor: "#4f46e5",
            color: "#ffffff",
            textDecoration: "none",
            borderRadius: "8px",
            fontSize: "13px",
            fontWeight: "600",
          }}
        >
          + Upload Invoices
        </Link>
      )}

      {hasFilters && totalDocs > 0 && (
        <div style={{ marginTop: "12px", fontSize: "12px", color: "#94a3b8" }}>
          There are {totalDocs} total documents in the database.
        </div>
      )}
    </div>
  );
}

// ── Document Details Modal (Inspect Drawer) ───────────────────────────────────

function DocumentDetailsModal({ doc, onClose, onExport, onDelete, copiedKey, onCopy }) {
  const [activeTab, setActiveTab] = useState("fields"); // 'fields' | 'json'

  const isLayoutLM =
    doc.extraction_method === "layoutlm" ||
    Boolean(doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0);

  const activeFields =
    isLayoutLM && doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0
      ? doc.layoutlm_fields
      : (doc.invoice_fields || doc.extracted_fields || {});

  const fileUrl = getDocumentFileUrl(doc.id);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(15, 23, 42, 0.6)",
        backdropFilter: "blur(4px)",
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "16px",
          width: "100%",
          maxWidth: "840px",
          maxHeight: "90vh",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 25px 50px -12px rgba(15, 23, 42, 0.25)",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid #e2e8f0",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            backgroundColor: "#f8fafc",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
              <span
                style={{
                  width: "28px",
                  height: "28px",
                  borderRadius: "6px",
                  backgroundColor: isLayoutLM ? "#f5f3ff" : "#f1f5f9",
                  color: isLayoutLM ? "#6d28d9" : "#475569",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <FileTextIcon size={15} />
              </span>
              <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#0f172a", margin: 0 }}>
                {doc.filename}
              </h2>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "12px", color: "#64748b" }}>
              <span>ID: <code style={{ backgroundColor: "#e2e8f0", padding: "1px 5px", borderRadius: "4px" }}>{doc.id}</code></span>
              <button
                onClick={() => onCopy(doc.id, "doc-id")}
                style={{
                  background: "none",
                  border: "none",
                  padding: 0,
                  color: "#4f46e5",
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                  fontSize: "11px",
                  fontWeight: "600",
                }}
              >
                <CopyIcon size={12} />
                {copiedKey === "doc-id" ? "Copied ID!" : "Copy ID"}
              </button>
            </div>
          </div>

          <button
            onClick={onClose}
            aria-label="Close Modal"
            style={{
              background: "none",
              border: "none",
              color: "#94a3b8",
              cursor: "pointer",
              padding: "6px",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <CloseIcon size={18} />
          </button>
        </div>

        {/* Action Toolbar */}
        <div
          style={{
            padding: "12px 24px",
            backgroundColor: "#ffffff",
            borderBottom: "1px solid #e2e8f0",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
          }}
        >
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              onClick={() => setActiveTab("fields")}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: "600",
                border: "none",
                backgroundColor: activeTab === "fields" ? "#eef2ff" : "transparent",
                color: activeTab === "fields" ? "#4f46e5" : "#64748b",
                cursor: "pointer",
              }}
            >
              Extracted Fields
            </button>
            <button
              onClick={() => setActiveTab("json")}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: "600",
                border: "none",
                backgroundColor: activeTab === "json" ? "#eef2ff" : "transparent",
                color: activeTab === "json" ? "#4f46e5" : "#64748b",
                cursor: "pointer",
              }}
            >
              Raw Metadata (JSON)
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <a
              href={fileUrl}
              target="_blank"
              rel="noreferrer"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 12px",
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
              Open Original
            </a>

            <button
              onClick={onExport}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 12px",
                fontSize: "12px",
                fontWeight: "600",
                color: "#334155",
                backgroundColor: "#f8fafc",
                border: "1px solid #cbd5e1",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              <DownloadIcon size={12} />
              Export
            </button>

            <Link
              to={`/review/${doc.id}`}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 14px",
                fontSize: "12px",
                fontWeight: "600",
                color: "#ffffff",
                backgroundColor: "#4f46e5",
                borderRadius: "6px",
                textDecoration: "none",
              }}
            >
              Review Page
              <ArrowRightIcon size={12} color="#ffffff" />
            </Link>

            <button
              onClick={onDelete}
              style={{
                padding: "6px 10px",
                backgroundColor: "#fee2e2",
                color: "#dc2626",
                border: "1px solid #fecaca",
                borderRadius: "6px",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
              }}
              title="Delete Document"
            >
              <TrashIcon size={13} color="#dc2626" />
            </button>
          </div>
        </div>

        {/* Modal Scrollable Body */}
        <div style={{ padding: "20px 24px", overflowY: "auto", flex: 1 }}>
          {/* Metadata Highlights Bar */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: "12px",
              padding: "14px 16px",
              backgroundColor: "#f8fafc",
              borderRadius: "10px",
              border: "1px solid #e2e8f0",
              marginBottom: "20px",
            }}
          >
            <div>
              <span style={{ fontSize: "11px", color: "#64748b", display: "block" }}>Extraction Method</span>
              <strong style={{ fontSize: "13px", color: isLayoutLM ? "#6d28d9" : "#334155" }}>
                {isLayoutLM ? "⚡ LayoutLMv3" : "📄 Baseline OCR"}
              </strong>
            </div>

            <div>
              <span style={{ fontSize: "11px", color: "#64748b", display: "block" }}>Processing</span>
              <strong style={{ fontSize: "13px", color: doc.processing_status === "processed" ? "#059669" : "#dc2626" }}>
                {doc.processing_status ? doc.processing_status.toUpperCase() : "PROCESSED"}
              </strong>
            </div>

            <div>
              <span style={{ fontSize: "11px", color: "#64748b", display: "block" }}>Verification</span>
              <strong style={{ fontSize: "13px", color: doc.verified ? "#15803d" : "#c2410c" }}>
                {doc.verified ? "✓ Verified" : "⚠ Pending"}
              </strong>
            </div>

            <div>
              <span style={{ fontSize: "11px", color: "#64748b", display: "block" }}>Confidence / Coverage</span>
              <strong style={{ fontSize: "13px", color: "#0f172a" }}>
                {doc.coverage_score ?? doc.confidence_score ?? 0}%
              </strong>
            </div>

            <div>
              <span style={{ fontSize: "11px", color: "#64748b", display: "block" }}>Pages</span>
              <strong style={{ fontSize: "13px", color: "#0f172a" }}>
                {doc.page_count || 1}
              </strong>
            </div>
          </div>

          {activeTab === "fields" ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              {/* Group 1: Parties */}
              <FieldSection title="Parties & Entities">
                <FieldDetailCard label="Vendor / Company Name" value={activeFields.vendor_name || activeFields.company_name} onCopy={onCopy} copyKey="modal-vendor" copiedKey={copiedKey} />
                <FieldDetailCard label="Customer Name" value={activeFields.customer_name} onCopy={onCopy} copyKey="modal-customer" copiedKey={copiedKey} />
                <FieldDetailCard label="Vendor GSTIN" value={activeFields.vendor_gstin || activeFields.gst_number} onCopy={onCopy} copyKey="modal-vgst" copiedKey={copiedKey} />
                <FieldDetailCard label="Customer GSTIN" value={activeFields.customer_gstin} onCopy={onCopy} copyKey="modal-cgst-num" copiedKey={copiedKey} />
              </FieldSection>

              {/* Group 2: Invoice Details */}
              <FieldSection title="Invoice Identifiers">
                <FieldDetailCard label="Invoice Number" value={activeFields.invoice_number} onCopy={onCopy} copyKey="modal-invnum" copiedKey={copiedKey} />
                <FieldDetailCard label="Invoice Date" value={activeFields.invoice_date} onCopy={onCopy} copyKey="modal-date" copiedKey={copiedKey} />
                <FieldDetailCard label="Due Date" value={activeFields.due_date} onCopy={onCopy} copyKey="modal-duedate" copiedKey={copiedKey} />
                <FieldDetailCard label="Purchase Order #" value={activeFields.po_number} onCopy={onCopy} copyKey="modal-po" copiedKey={copiedKey} />
              </FieldSection>

              {/* Group 3: Financials & Tax */}
              <FieldSection title="Financial & Tax Breakdown">
                <FieldDetailCard label="Subtotal" value={activeFields.subtotal ? `${activeFields.currency || "₹"} ${activeFields.subtotal}` : undefined} onCopy={onCopy} copyKey="modal-subtotal" copiedKey={copiedKey} />
                <FieldDetailCard label="CGST" value={activeFields.cgst ? `${activeFields.currency || "₹"} ${activeFields.cgst}` : undefined} onCopy={onCopy} copyKey="modal-cgst" copiedKey={copiedKey} />
                <FieldDetailCard label="SGST" value={activeFields.sgst ? `${activeFields.currency || "₹"} ${activeFields.sgst}` : undefined} onCopy={onCopy} copyKey="modal-sgst" copiedKey={copiedKey} />
                <FieldDetailCard label="Total Tax" value={activeFields.total_tax ? `${activeFields.currency || "₹"} ${activeFields.total_tax}` : undefined} onCopy={onCopy} copyKey="modal-tax" copiedKey={copiedKey} />
                <FieldDetailCard label="Total Amount" value={activeFields.total_amount ? `${activeFields.currency || "₹"} ${activeFields.total_amount}` : undefined} highlight onCopy={onCopy} copyKey="modal-total" copiedKey={copiedKey} />
                <FieldDetailCard label="Currency" value={activeFields.currency} onCopy={onCopy} copyKey="modal-curr" copiedKey={copiedKey} />
              </FieldSection>
            </div>
          ) : (
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "8px",
                }}
              >
                <span style={{ fontSize: "12px", color: "#64748b", fontWeight: "600" }}>Raw Document Metadata Object:</span>
                <button
                  onClick={() => onCopy(JSON.stringify(doc, null, 2), "modal-json")}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#4f46e5",
                    fontSize: "12px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "4px",
                  }}
                >
                  <CopyIcon size={12} />
                  {copiedKey === "modal-json" ? "Copied JSON!" : "Copy Full JSON"}
                </button>
              </div>
              <pre
                style={{
                  backgroundColor: "#0f172a",
                  color: "#f8fafc",
                  padding: "16px",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontFamily: "monospace",
                  overflowX: "auto",
                  maxHeight: "360px",
                }}
              >
                {JSON.stringify(doc, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "16px 24px",
            borderTop: "1px solid #e2e8f0",
            backgroundColor: "#f8fafc",
            display: "flex",
            justifyContent: "flex-end",
            gap: "10px",
          }}
        >
          <button
            onClick={onClose}
            style={{
              padding: "8px 16px",
              backgroundColor: "#ffffff",
              border: "1px solid #cbd5e1",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              color: "#475569",
              cursor: "pointer",
            }}
          >
            Close
          </button>
          <Link
            to={`/review/${doc.id}`}
            style={{
              padding: "8px 18px",
              backgroundColor: "#4f46e5",
              color: "#ffffff",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            Open in Document Review
            <ArrowRightIcon size={12} color="#ffffff" />
          </Link>
        </div>
      </div>
    </div>
  );
}

function FieldSection({ title, children }) {
  return (
    <div>
      <h3 style={{ fontSize: "13px", fontWeight: "700", color: "#334155", textTransform: "uppercase", letterSpacing: "0.04em", margin: "0 0 10px" }}>
        {title}
      </h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "10px" }}>
        {children}
      </div>
    </div>
  );
}

function FieldDetailCard({ label, value, highlight = false, onCopy, copyKey, copiedKey }) {
  const displayVal = value && value !== "Not found" ? String(value) : "—";
  const isAvailable = displayVal !== "—";

  return (
    <div
      style={{
        padding: "10px 14px",
        borderRadius: "8px",
        backgroundColor: highlight ? "#f5f3ff" : "#f8fafc",
        border: highlight ? "1px solid #ddd6fe" : "1px solid #e2e8f0",
        position: "relative",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
        <span style={{ fontSize: "11px", color: highlight ? "#6d28d9" : "#64748b", fontWeight: "600" }}>
          {label}
        </span>
        {isAvailable && onCopy && (
          <button
            onClick={() => onCopy(displayVal, copyKey)}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              color: "#94a3b8",
              cursor: "pointer",
              fontSize: "10px",
              display: "flex",
              alignItems: "center",
              gap: "2px",
            }}
            title="Copy value"
          >
            <CopyIcon size={11} />
            {copiedKey === copyKey ? "Copied" : ""}
          </button>
        )}
      </div>

      <div
        style={{
          fontSize: highlight ? "15px" : "13px",
          fontWeight: highlight ? "700" : "600",
          color: isAvailable ? (highlight ? "#4f46e5" : "#0f172a") : "#94a3b8",
          wordBreak: "break-all",
        }}
      >
        {displayVal}
      </div>
    </div>
  );
}

// ── Delete Confirmation Dialog Modal ──────────────────────────────────────────

function DeleteConfirmationModal({ filename, isDeleting, onConfirm, onCancel }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(15, 23, 42, 0.65)",
        backdropFilter: "blur(4px)",
        zIndex: 1100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
    >
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "14px",
          width: "100%",
          maxWidth: "460px",
          padding: "24px",
          boxShadow: "0 25px 50px -12px rgba(15, 23, 42, 0.3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "14px", marginBottom: "16px" }}>
          <div
            style={{
              width: "42px",
              height: "42px",
              borderRadius: "50%",
              backgroundColor: "#fee2e2",
              color: "#dc2626",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <TrashIcon size={20} color="#dc2626" />
          </div>

          <div>
            <h3 style={{ fontSize: "17px", fontWeight: "700", color: "#0f172a", margin: "0 0 6px" }}>
              Delete Document Record
            </h3>
            <p style={{ fontSize: "13px", color: "#64748b", margin: 0, lineHeight: 1.5 }}>
              Are you sure you want to permanently delete <strong>{filename}</strong>? This action will remove the uploaded file, extracted field records, and all audit logs from the database.
            </p>
          </div>
        </div>

        <div
          style={{
            padding: "10px 14px",
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: "8px",
            color: "#991b1b",
            fontSize: "12px",
            marginBottom: "20px",
          }}
        >
          <strong>Warning:</strong> This deletion is permanent and cannot be undone.
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
          <button
            onClick={onCancel}
            disabled={isDeleting}
            style={{
              padding: "9px 16px",
              backgroundColor: "#ffffff",
              border: "1px solid #cbd5e1",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              color: "#475569",
              cursor: isDeleting ? "not-allowed" : "pointer",
            }}
          >
            Cancel
          </button>

          <button
            onClick={onConfirm}
            disabled={isDeleting}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "9px 18px",
              backgroundColor: "#dc2626",
              border: "none",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              color: "#ffffff",
              cursor: isDeleting ? "not-allowed" : "pointer",
              boxShadow: "0 1px 2px rgba(220, 38, 38, 0.2)",
            }}
          >
            {isDeleting ? "Deleting..." : "Delete Document"}
          </button>
        </div>
      </div>
    </div>
  );
}
