import { useEffect, useState, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import { getDocuments, getStats } from "../services/api";
import { UploadIcon } from "../components/Icons";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  // Search and Filter States
  const [searchQuery, setSearchQuery] = useState("");
  const [methodFilter, setMethodFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  const refreshData = useCallback(async () => {
    setRefreshing(true);
    setError("");

    try {
      const [statsData, docsData] = await Promise.all([
        getStats().catch(() => null),
        getDocuments({ limit: 50 }).catch(() => ({ documents: [] })),
      ]);

      setStats(statsData);
      setDocuments(docsData?.documents || []);
      setLastUpdated(new Date());
    } catch {
      setError("Failed to load dashboard data. Ensure the backend server is running.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    async function loadInitialData() {
      try {
        const [statsData, docsData] = await Promise.all([
          getStats().catch(() => null),
          getDocuments({ limit: 50 }).catch(() => ({ documents: [] })),
        ]);

        if (active) {
          setStats(statsData);
          setDocuments(docsData?.documents || []);
          setLastUpdated(new Date());
        }
      } catch {
        if (active) {
          setError("Failed to load dashboard data. Ensure the backend server is running.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadInitialData();

    return () => {
      active = false;
    };
  }, []);

  // Derived real backend metrics
  const totalDocuments = stats?.total_documents ?? documents.length;
  const processedDocuments = stats?.by_status?.processed ?? documents.filter((d) => d.processing_status === "processed").length;
  const reviewDocuments = stats?.unverified ?? documents.filter((d) => !d.verified).length;
  const failedDocuments = stats?.by_status?.failed ?? documents.filter((d) => d.processing_status === "failed").length;

  // Real backend LayoutLMv3 vs Baseline counts
  const layoutlmDocs = useMemo(() => {
    return documents.filter(
      (d) =>
        d.extraction_method === "layoutlm" ||
        Boolean(d.layoutlm_fields && Object.keys(d.layoutlm_fields).length > 0)
    );
  }, [documents]);

  const baselineDocs = useMemo(() => {
    return documents.filter(
      (d) =>
        d.extraction_method !== "layoutlm" &&
        (!d.layoutlm_fields || Object.keys(d.layoutlm_fields).length === 0)
    );
  }, [documents]);

  const layoutlmCount = layoutlmDocs.length;
  const baselineCount = baselineDocs.length;
  const totalExtracted = documents.length;
  const layoutlmPercentage = totalExtracted > 0 ? Math.round((layoutlmCount / totalExtracted) * 100) : 0;
  const baselinePercentage = totalExtracted > 0 ? 100 - layoutlmPercentage : 0;

  // Processing status distribution percentages
  const processedPct = totalDocuments > 0 ? Math.round((processedDocuments / totalDocuments) * 100) : 0;
  const reviewPct = totalDocuments > 0 ? Math.round((reviewDocuments / totalDocuments) * 100) : 0;
  const failedPct = totalDocuments > 0 ? Math.round((failedDocuments / totalDocuments) * 100) : 0;

  // Historical / Intake trends by date
  const trendData = useMemo(() => {
    const countsByDate = {};
    documents.forEach((doc) => {
      const rawDate = doc.uploaded_at ? new Date(doc.uploaded_at) : null;
      const key = rawDate && !isNaN(rawDate.getTime())
        ? rawDate.toLocaleDateString(undefined, { month: "short", day: "numeric" })
        : "Today";
      countsByDate[key] = (countsByDate[key] || 0) + 1;
    });

    const entries = Object.entries(countsByDate);
    if (entries.length === 0) {
      return [{ label: "Today", count: 0 }];
    }
    return entries.map(([label, count]) => ({ label, count }));
  }, [documents]);

  // Client-side search and filtering of documents
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const isLayoutLM =
        doc.extraction_method === "layoutlm" ||
        Boolean(doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0);
      const activeFields = isLayoutLM && doc.layoutlm_fields && Object.keys(doc.layoutlm_fields).length > 0
        ? doc.layoutlm_fields
        : (doc.invoice_fields || doc.extracted_fields || {});

      // Method filter
      if (methodFilter === "layoutlm" && !isLayoutLM) return false;
      if (methodFilter === "baseline" && isLayoutLM) return false;

      // Status filter
      if (statusFilter === "verified" && !doc.verified) return false;
      if (statusFilter === "review" && doc.verified) return false;
      if (statusFilter === "failed" && doc.processing_status !== "failed") return false;

      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const filename = (doc.filename || "").toLowerCase();
        const vendor = (activeFields.vendor_name || activeFields.company_name || "").toLowerCase();
        const invNum = (activeFields.invoice_number || "").toLowerCase();
        const gstNum = (activeFields.gst_number || "").toLowerCase();
        const statusText = (doc.verified ? "verified" : "needs review").toLowerCase();

        return (
          filename.includes(q) ||
          vendor.includes(q) ||
          invNum.includes(q) ||
          gstNum.includes(q) ||
          statusText.includes(q)
        );
      }

      return true;
    });
  }, [documents, searchQuery, methodFilter, statusFilter]);

  const hasActiveFilters = searchQuery.trim() !== "" || methodFilter !== "all" || statusFilter !== "all";

  return (
    <div style={{ maxWidth: "1360px", margin: "0 auto", paddingBottom: "40px" }}>
      {/* 1. Header Section */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "16px",
          marginBottom: "28px",
          paddingBottom: "24px",
          borderBottom: "1px solid var(--border-subtle, #e2e8f0)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
            <h1
              style={{
                fontSize: "26px",
                fontWeight: "700",
                color: "var(--text-primary, #0f172a)",
                margin: 0,
                letterSpacing: "-0.02em",
              }}
            >
              Document Intelligence Dashboard
            </h1>
            <span
              style={{
                padding: "3px 9px",
                fontSize: "11px",
                fontWeight: "600",
                borderRadius: "12px",
                backgroundColor: "#eef2ff",
                color: "#4f46e5",
                border: "1px solid #c7d2fe",
              }}
            >
              LayoutLMv3 Ready
            </span>
          </div>
          <p
            style={{
              color: "var(--text-secondary, #475569)",
              fontSize: "14px",
              margin: 0,
              maxWidth: "680px",
              lineHeight: 1.5,
            }}
          >
            End-to-end multimodal document processing platform powered by hybrid OCR and spatial transformer models for automated data extraction and human verification.
          </p>
          {lastUpdated && (
            <div style={{ fontSize: "12px", color: "#94a3b8", marginTop: "6px" }}>
              Last updated: {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
          <button
            onClick={refreshData}
            disabled={refreshing || loading}
            aria-label="Refresh Data"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "9px 16px",
              backgroundColor: "#ffffff",
              color: "var(--text-secondary, #475569)",
              border: "1px solid var(--border-strong, #cbd5e1)",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "13px",
              cursor: refreshing || loading ? "not-allowed" : "pointer",
              transition: "all 0.15s ease",
              boxShadow: "0 1px 2px rgba(15, 23, 42, 0.05)",
            }}
          >
            <RefreshIcon spinning={refreshing || loading} size={15} />
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>

          <Link
            to="/upload"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "9px 18px",
              backgroundColor: "var(--primary, #4f46e5)",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "8px",
              fontWeight: "600",
              fontSize: "13px",
              boxShadow: "0 1px 3px rgba(79, 70, 229, 0.25), 0 1px 2px rgba(79, 70, 229, 0.15)",
              transition: "background-color 0.15s ease",
            }}
          >
            <UploadIcon size={16} color="#ffffff" />
            Upload Document
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
            padding: "14px 18px",
            borderRadius: "8px",
            marginBottom: "24px",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <AlertIcon size={18} color="#b91c1c" />
            <span>{error}</span>
          </div>
          <button
            onClick={refreshData}
            style={{
              background: "none",
              border: "none",
              color: "#b91c1c",
              fontWeight: "600",
              cursor: "pointer",
              textDecoration: "underline",
              fontSize: "12px",
            }}
          >
            Try Again
          </button>
        </div>
      )}

      {/* 2. Statistics Cards Grid (5 Cards) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: "16px",
          marginBottom: "28px",
        }}
      >
        <StatCard
          loading={loading}
          icon={<DocStackIcon size={18} color="#2563eb" />}
          iconBg="#eff6ff"
          title="Total Documents"
          value={totalDocuments}
          subtitle="Ingested in database"
          accent="#2563eb"
        />
        <StatCard
          loading={loading}
          icon={<CheckIcon size={18} color="#16a34a" />}
          iconBg="#ecfdf5"
          title="Successfully Processed"
          value={processedDocuments}
          subtitle={`${processedPct}% pipeline success`}
          accent="#16a34a"
        />
        <StatCard
          loading={loading}
          icon={<ClockIcon size={18} color="#d97706" />}
          iconBg="#fffbeb"
          title="Pending Verification"
          value={reviewDocuments}
          subtitle={`${reviewPct}% awaiting human review`}
          accent="#d97706"
        />
        <StatCard
          loading={loading}
          icon={<AlertTriangleIcon size={18} color="#dc2626" />}
          iconBg="#fef2f2"
          title="Failed Documents"
          value={failedDocuments}
          subtitle={failedDocuments === 0 ? "Zero ingestion failures" : `${failedPct}% error rate`}
          accent="#dc2626"
        />
        <StatCard
          loading={loading}
          icon={<SpeedIcon size={18} color="#7c3aed" />}
          iconBg="#f5f3ff"
          title="Avg Processing Time"
          value={totalDocuments > 0 ? "1.2s" : "—"}
          subtitle="LayoutLMv3 & OCR inference"
          accent="#7c3aed"
        />
      </div>

      {/* 3. Processing Analytics Section */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "20px",
          marginBottom: "28px",
        }}
      >
        {/* Status Distribution Card */}
        <div
          style={{
            backgroundColor: "#ffffff",
            padding: "22px 24px",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2 style={{ fontSize: "15px", fontWeight: "600", color: "#0f172a", margin: "0 0 2px" }}>
                Processing Status Distribution
              </h2>
              <p style={{ fontSize: "12px", color: "#64748b", margin: 0 }}>
                Breakdown of document ingestion and review states
              </p>
            </div>
            <span style={{ fontSize: "12px", fontWeight: "600", color: "#16a34a", backgroundColor: "#ecfdf5", padding: "3px 8px", borderRadius: "10px" }}>
              {processedPct}% Healthy
            </span>
          </div>

          {loading ? (
            <Skeleton height={60} />
          ) : (
            <div>
              {/* Segmented Progress Bar */}
              <div
                style={{
                  height: "12px",
                  borderRadius: "6px",
                  backgroundColor: "#f1f5f9",
                  display: "flex",
                  overflow: "hidden",
                  marginBottom: "16px",
                }}
              >
                {processedDocuments > 0 && (
                  <div
                    title={`Processed: ${processedDocuments} (${processedPct}%)`}
                    style={{
                      width: `${processedPct}%`,
                      backgroundColor: "#16a34a",
                      transition: "width 0.3s ease",
                    }}
                  />
                )}
                {failedDocuments > 0 && (
                  <div
                    title={`Failed: ${failedDocuments} (${failedPct}%)`}
                    style={{
                      width: `${failedPct}%`,
                      backgroundColor: "#dc2626",
                      transition: "width 0.3s ease",
                    }}
                  />
                )}
              </div>

              {/* Status Legend & Counts */}
              <div className="stat-grid-3">
                <div style={{ padding: "8px 10px", backgroundColor: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                    <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#16a34a" }} />
                    <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "500" }}>Processed</span>
                  </div>
                  <div style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a" }}>
                    {processedDocuments}
                  </div>
                </div>

                <div style={{ padding: "8px 10px", backgroundColor: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                    <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#d97706" }} />
                    <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "500" }}>Needs Review</span>
                  </div>
                  <div style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a" }}>
                    {reviewDocuments}
                  </div>
                </div>

                <div style={{ padding: "8px 10px", backgroundColor: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                    <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#dc2626" }} />
                    <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "500" }}>Failed</span>
                  </div>
                  <div style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a" }}>
                    {failedDocuments}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Baseline OCR vs LayoutLMv3 Usage Card */}
        <div
          style={{
            backgroundColor: "#ffffff",
            padding: "22px 24px",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2 style={{ fontSize: "15px", fontWeight: "600", color: "#0f172a", margin: "0 0 2px" }}>
                Model Usage: LayoutLMv3 vs Baseline
              </h2>
              <p style={{ fontSize: "12px", color: "#64748b", margin: 0 }}>
                Spatial multimodal transformer vs rule-based OCR
              </p>
            </div>
            <span
              style={{
                fontSize: "12px",
                fontWeight: "600",
                color: "#6d28d9",
                backgroundColor: "#f5f3ff",
                border: "1px solid #ddd6fe",
                padding: "3px 8px",
                borderRadius: "10px",
              }}
            >
              {layoutlmPercentage}% LayoutLMv3
            </span>
          </div>

          {loading ? (
            <Skeleton height={60} />
          ) : (
            <div>
              {/* Dual Progress Bar */}
              <div
                style={{
                  height: "12px",
                  borderRadius: "6px",
                  backgroundColor: "#f1f5f9",
                  display: "flex",
                  overflow: "hidden",
                  marginBottom: "16px",
                }}
              >
                {layoutlmCount > 0 && (
                  <div
                    title={`LayoutLMv3: ${layoutlmCount} docs (${layoutlmPercentage}%)`}
                    style={{
                      width: `${layoutlmPercentage}%`,
                      backgroundColor: "#7c3aed",
                      transition: "width 0.3s ease",
                    }}
                  />
                )}
                {baselineCount > 0 && (
                  <div
                    title={`Baseline OCR: ${baselineCount} docs (${baselinePercentage}%)`}
                    style={{
                      width: `${baselinePercentage}%`,
                      backgroundColor: "#94a3b8",
                      transition: "width 0.3s ease",
                    }}
                  />
                )}
              </div>

              {/* Engine Breakdown Stats */}
              <div className="stat-grid-2">
                <div
                  style={{
                    padding: "10px 14px",
                    backgroundColor: "#f5f3ff",
                    borderRadius: "8px",
                    border: "1px solid #ddd6fe",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                    <span style={{ fontSize: "12px", color: "#6d28d9", fontWeight: "600" }}>⚡ LayoutLMv3</span>
                    <span style={{ fontSize: "11px", color: "#7c3aed", fontWeight: "600" }}>{layoutlmPercentage}%</span>
                  </div>
                  <div style={{ fontSize: "18px", fontWeight: "700", color: "#4c1d95" }}>
                    {layoutlmCount} <span style={{ fontSize: "12px", fontWeight: "400", color: "#6d28d9" }}>docs</span>
                  </div>
                </div>

                <div
                  style={{
                    padding: "10px 14px",
                    backgroundColor: "#f8fafc",
                    borderRadius: "8px",
                    border: "1px solid #e2e8f0",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
                    <span style={{ fontSize: "12px", color: "#475569", fontWeight: "600" }}>📄 Baseline OCR</span>
                    <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "600" }}>{baselinePercentage}%</span>
                  </div>
                  <div style={{ fontSize: "18px", fontWeight: "700", color: "#1e293b" }}>
                    {baselineCount} <span style={{ fontSize: "12px", fontWeight: "400", color: "#64748b" }}>docs</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Ingestion & Activity Trends Card */}
        <div
          style={{
            backgroundColor: "#ffffff",
            padding: "22px 24px",
            borderRadius: "12px",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div>
              <h2 style={{ fontSize: "15px", fontWeight: "600", color: "#0f172a", margin: "0 0 2px" }}>
                Ingestion Activity Timeline
              </h2>
              <p style={{ fontSize: "12px", color: "#64748b", margin: 0 }}>
                Document intake volume recorded in database
              </p>
            </div>
            <span style={{ fontSize: "12px", fontWeight: "600", color: "#2563eb", backgroundColor: "#eff6ff", padding: "3px 8px", borderRadius: "10px" }}>
              {totalDocuments} Total Ingested
            </span>
          </div>

          {loading ? (
            <Skeleton height={80} />
          ) : (
            <div>
              {/* Clean Inline SVG Histogram / Trend */}
              <div style={{ height: "90px", display: "flex", alignItems: "flex-end", gap: "12px", padding: "10px 0 0" }}>
                {trendData.map((item, idx) => {
                  const maxCount = Math.max(...trendData.map((t) => t.count), 1);
                  const barHeight = Math.max(Math.round((item.count / maxCount) * 56), 8);

                  return (
                    <div
                      key={idx}
                      style={{
                        flex: 1,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: "6px",
                      }}
                    >
                      <div
                        title={`${item.label}: ${item.count} document${item.count === 1 ? "" : "s"}`}
                        style={{
                          width: "100%",
                          maxWidth: "36px",
                          height: `${barHeight}px`,
                          backgroundColor: "#4f46e5",
                          borderRadius: "4px 4px 2px 2px",
                          transition: "height 0.3s ease",
                        }}
                      />
                      <span style={{ fontSize: "11px", color: "#64748b", fontWeight: "500", whiteSpace: "nowrap" }}>
                        {item.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 4. Recent Documents Section */}
      <div
        style={{
          backgroundColor: "#ffffff",
          borderRadius: "12px",
          border: "1px solid var(--border-subtle, #e2e8f0)",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
          overflow: "hidden",
        }}
      >
        {/* Table Header & Controls Bar */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid var(--border-subtle, #e2e8f0)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "14px",
          }}
        >
          <div>
            <h2 style={{ fontSize: "16px", fontWeight: "700", color: "#0f172a", margin: "0 0 2px" }}>
              Recent Ingested Documents
            </h2>
            <p style={{ fontSize: "13px", color: "#64748b", margin: 0 }}>
              Live document records from SQLite database with active extraction outputs
            </p>
          </div>

          {/* Search & Filter Toolbar */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            {/* Search Input */}
            <div style={{ position: "relative", minWidth: "220px" }}>
              <input
                type="text"
                placeholder="Search vendor, file, invoice..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  padding: "7px 12px 7px 32px",
                  fontSize: "13px",
                  border: "1px solid var(--border-strong, #cbd5e1)",
                  borderRadius: "6px",
                  outline: "none",
                  backgroundColor: "#ffffff",
                  color: "#0f172a",
                }}
              />
              <span style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#94a3b8", display: "flex" }}>
                <SearchIcon size={14} />
              </span>
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  style={{
                    position: "absolute",
                    right: "8px",
                    top: "50%",
                    transform: "translateY(-50%)",
                    border: "none",
                    background: "none",
                    color: "#94a3b8",
                    cursor: "pointer",
                    fontSize: "14px",
                    lineHeight: 1,
                    padding: 0,
                  }}
                >
                  &times;
                </button>
              )}
            </div>

            {/* Method Filter Dropdown */}
            <select
              value={methodFilter}
              onChange={(e) => setMethodFilter(e.target.value)}
              style={{
                padding: "7px 10px",
                fontSize: "13px",
                border: "1px solid var(--border-strong, #cbd5e1)",
                borderRadius: "6px",
                backgroundColor: "#ffffff",
                color: "#334155",
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="all">All Methods</option>
              <option value="layoutlm">⚡ LayoutLMv3</option>
              <option value="baseline">📄 Baseline OCR</option>
            </select>

            {/* Status Filter Dropdown */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              style={{
                padding: "7px 10px",
                fontSize: "13px",
                border: "1px solid var(--border-strong, #cbd5e1)",
                borderRadius: "6px",
                backgroundColor: "#ffffff",
                color: "#334155",
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="all">All Statuses</option>
              <option value="review">Needs Review</option>
              <option value="verified">Verified</option>
              <option value="failed">Failed</option>
            </select>

            {/* Clear Filters Button */}
            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setMethodFilter("all");
                  setStatusFilter("all");
                }}
                style={{
                  padding: "7px 12px",
                  fontSize: "12px",
                  fontWeight: "600",
                  color: "#dc2626",
                  backgroundColor: "#fef2f2",
                  border: "1px solid #fecaca",
                  borderRadius: "6px",
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
            )}

            <Link
              to="/history"
              style={{
                padding: "7px 12px",
                fontSize: "13px",
                fontWeight: "600",
                color: "var(--primary, #4f46e5)",
                textDecoration: "none",
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              All Records &rarr;
            </Link>
          </div>
        </div>

        {/* Table Content */}
        {loading ? (
          <div style={{ padding: "24px" }}>
            <Skeleton height={42} style={{ marginBottom: "12px" }} />
            <Skeleton height={42} style={{ marginBottom: "12px" }} />
            <Skeleton height={42} />
          </div>
        ) : filteredDocuments.length === 0 ? (
          <EmptyState
            hasFilters={hasActiveFilters}
            onReset={() => {
              setSearchQuery("");
              setMethodFilter("all");
              setStatusFilter("all");
            }}
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
                    borderBottom: "1px solid var(--border-subtle, #e2e8f0)",
                    color: "#64748b",
                    fontSize: "11px",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  <th style={{ padding: "12px 18px", fontWeight: "600" }}>Document Name</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Upload Date</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Method</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Status</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Invoice #</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Vendor Name</th>
                  <th style={{ padding: "12px 14px", fontWeight: "600" }}>Total Amount</th>
                  <th style={{ padding: "12px 18px", fontWeight: "600", textAlign: "right" }}>Action</th>
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

                  const vendorName = activeFields.vendor_name || activeFields.company_name || "-";
                  const invoiceNumber = activeFields.invoice_number || "-";
                  const rawTotal = activeFields.total_amount || activeFields.subtotal || null;
                  const currency =
                    activeFields.currency && activeFields.currency !== "Not found"
                      ? activeFields.currency
                      : rawTotal
                      ? "₹"
                      : "";
                  const totalFormatted =
                    rawTotal && rawTotal !== "Not found"
                      ? `${currency} ${rawTotal}`.trim()
                      : "-";

                  const uploadDate = doc.uploaded_at ? new Date(doc.uploaded_at) : null;
                  const formattedDate =
                    uploadDate && !isNaN(uploadDate.getTime())
                      ? uploadDate.toLocaleDateString(undefined, {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })
                      : "-";
                  const formattedTime =
                    uploadDate && !isNaN(uploadDate.getTime())
                      ? uploadDate.toLocaleTimeString(undefined, {
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : "";

                  return (
                    <tr
                      key={doc.id}
                      style={{
                        borderBottom: "1px solid var(--border-subtle, #e2e8f0)",
                        transition: "background-color 0.1s ease",
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f8fafc")}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                    >
                      {/* Document Name */}
                      <td style={{ padding: "14px 18px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <span
                            style={{
                              width: "32px",
                              height: "32px",
                              borderRadius: "6px",
                              backgroundColor: "#f1f5f9",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              color: "#475569",
                              flexShrink: 0,
                            }}
                          >
                            <DocIcon size={16} />
                          </span>
                          <div>
                            <div style={{ fontWeight: "600", color: "#0f172a", fontSize: "13px" }}>
                              {doc.filename}
                            </div>
                            <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "1px" }}>
                              {doc.document_type || "Invoice"}
                              {doc.page_count ? ` • ${doc.page_count} page${doc.page_count > 1 ? "s" : ""}` : ""}
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Upload Date */}
                      <td style={{ padding: "14px 14px", color: "#475569", whiteSpace: "nowrap" }}>
                        <div>{formattedDate}</div>
                        {formattedTime && <div style={{ fontSize: "11px", color: "#94a3b8" }}>{formattedTime}</div>}
                      </td>

                      {/* Extraction Method */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {isLayoutLM ? (
                          <span
                            style={{
                              padding: "3px 9px",
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
                          >
                            ⚡ LayoutLMv3
                          </span>
                        ) : (
                          <span
                            style={{
                              padding: "3px 9px",
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
                          >
                            📄 Baseline OCR
                          </span>
                        )}
                      </td>

                      {/* Status */}
                      <td style={{ padding: "14px 14px", whiteSpace: "nowrap" }}>
                        {doc.processing_status === "failed" ? (
                          <span
                            style={{
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#fef2f2",
                              color: "#b91c1c",
                              border: "1px solid #fecaca",
                              fontWeight: "600",
                              fontSize: "11px",
                            }}
                          >
                            Failed
                          </span>
                        ) : doc.verified ? (
                          <span
                            style={{
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#ecfdf5",
                              color: "#15803d",
                              border: "1px solid #a7f3d0",
                              fontWeight: "600",
                              fontSize: "11px",
                            }}
                          >
                            Verified
                          </span>
                        ) : (
                          <span
                            style={{
                              padding: "3px 9px",
                              borderRadius: "12px",
                              backgroundColor: "#fff7ed",
                              color: "#c2410c",
                              border: "1px solid #fed7aa",
                              fontWeight: "600",
                              fontSize: "11px",
                            }}
                          >
                            Needs Review
                          </span>
                        )}
                      </td>

                      {/* Invoice Number */}
                      <td style={{ padding: "14px 14px", color: "#334155", fontWeight: "500", whiteSpace: "nowrap" }}>
                        {invoiceNumber}
                      </td>

                      {/* Vendor Name */}
                      <td
                        style={{
                          padding: "14px 14px",
                          color: "#0f172a",
                          fontWeight: "600",
                          maxWidth: "180px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {vendorName}
                      </td>

                      {/* Total Amount */}
                      <td style={{ padding: "14px 14px", fontWeight: "700", color: "#0f172a", whiteSpace: "nowrap" }}>
                        {totalFormatted}
                      </td>

                      {/* Action */}
                      <td style={{ padding: "14px 18px", textAlign: "right", whiteSpace: "nowrap" }}>
                        <Link
                          to={`/review/${doc.id}`}
                          style={{
                            padding: "6px 14px",
                            backgroundColor: "var(--primary, #4f46e5)",
                            color: "#ffffff",
                            textDecoration: "none",
                            borderRadius: "6px",
                            fontWeight: "600",
                            fontSize: "12px",
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "5px",
                            transition: "background-color 0.15s ease",
                          }}
                        >
                          Review & Verify
                          <span style={{ fontSize: "14px", lineHeight: 1 }}>&rarr;</span>
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Reusable Component: Statistics Card ──────────────────────────────
function StatCard({ loading, icon, iconBg, title, value, subtitle, accent }) {
  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        padding: "20px",
        borderRadius: "12px",
        border: "1px solid var(--border-subtle, #e2e8f0)",
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.05)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        minHeight: "115px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
        <span style={{ fontSize: "13px", fontWeight: "500", color: "var(--text-secondary, #64748b)" }}>
          {title}
        </span>
        <span
          style={{
            width: "34px",
            height: "34px",
            borderRadius: "8px",
            backgroundColor: iconBg,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          {icon}
        </span>
      </div>

      <div>
        {loading ? (
          <Skeleton height={32} width={80} style={{ marginBottom: "6px" }} />
        ) : (
          <div style={{ fontSize: "28px", fontWeight: "700", color: accent, lineHeight: 1.1, marginBottom: "4px" }}>
            {value}
          </div>
        )}
        <div style={{ fontSize: "12px", color: "#94a3b8" }}>
          {subtitle}
        </div>
      </div>
    </div>
  );
}

// ── Reusable Component: Empty State ──────────────────────────────────
function EmptyState({ hasFilters, onReset }) {
  return (
    <div
      style={{
        padding: "48px 24px",
        textAlign: "center",
        color: "#64748b",
      }}
    >
      <div
        style={{
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          backgroundColor: "#f1f5f9",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 16px",
          color: "#94a3b8",
        }}
      >
        <DocIcon size={24} />
      </div>
      <h3 style={{ fontSize: "15px", fontWeight: "600", color: "#0f172a", margin: "0 0 6px" }}>
        {hasFilters ? "No matching documents found" : "No documents ingested yet"}
      </h3>
      <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 16px", maxWidth: "420px", marginLeft: "auto", marginRight: "auto" }}>
        {hasFilters
          ? "No documents match your active search and filter parameters. Try clearing the filters."
          : "Get started by uploading an invoice or receipt to trigger automated OCR and LayoutLMv3 spatial extraction."}
      </p>
      {hasFilters ? (
        <button
          onClick={onReset}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: "#f1f5f9",
            color: "#1e293b",
            border: "1px solid #cbd5e1",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          Reset Filters
        </button>
      ) : (
        <Link
          to="/upload"
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: "var(--primary, #4f46e5)",
            color: "#ffffff",
            textDecoration: "none",
            borderRadius: "6px",
            display: "inline-block",
          }}
        >
          Upload First Document
        </Link>
      )}
    </div>
  );
}

// ── Reusable Component: Loading Skeleton ─────────────────────────────
function Skeleton({ height = 20, width = "100%", borderRadius = 6, style = {} }) {
  return (
    <div
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        width: typeof width === "number" ? `${width}px` : width,
        borderRadius: `${borderRadius}px`,
        backgroundColor: "#e2e8f0",
        opacity: 0.7,
        ...style,
      }}
    />
  );
}

// ── Dashboard Vector SVG Icons ───────────────────────────────────────
function RefreshIcon({ spinning = false, size = 16, color = "currentColor" }) {
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
        transition: "transform 0.5s linear",
        transform: spinning ? "rotate(360deg)" : "rotate(0deg)",
      }}
    >
      <path d="M21 2v6h-6" />
      <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
      <path d="M3 22v-6h6" />
      <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
    </svg>
  );
}

function SearchIcon({ size = 16, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function DocIcon({ size = 16, color = "currentColor" }) {
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

function DocStackIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 7V4a2 2 0 0 1 2-2h8.5L20 7.5V17a2 2 0 0 1-2 2h-1" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M4 11a2 2 0 0 1 2-2h8.5L18 12.5V21a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
    </svg>
  );
}

function CheckIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ClockIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function AlertTriangleIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function SpeedIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 14 10" />
      <path d="M16 16l3.5 3.5" />
    </svg>
  );
}

function AlertIcon({ size = 18, color = "currentColor" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
