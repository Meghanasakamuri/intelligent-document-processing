import { useState, useEffect } from "react";
import { NavLink, Link, Outlet, useLocation } from "react-router-dom";
import {
  DashboardIcon,
  UploadIcon,
  ReviewIcon,
  HistoryIcon,
  EvaluationIcon,
  QAIcon,
  MenuIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  SparklesIcon,
  LogoIcon,
} from "./Icons";

const NAV_ITEMS = [
  {
    category: "INGESTION & WORKFLOW",
    items: [
      { to: "/", label: "Dashboard", icon: DashboardIcon, end: true },
      { to: "/upload", label: "Upload Documents", icon: UploadIcon },
      { to: "/review", label: "Document Review", icon: ReviewIcon },
      { to: "/history", label: "Document History", icon: HistoryIcon },
    ],
  },
  {
    category: "RESEARCH & MODELS",
    items: [
      { to: "/evaluation", label: "Evaluation & Analytics", icon: EvaluationIcon },
      { to: "/qa", label: "Document QA", icon: QAIcon },
    ],
  },
];

function getRouteMetadata(pathname) {
  if (pathname === "/") {
    return {
      title: "System Performance Dashboard",
      breadcrumbs: ["Platform", "Dashboard"],
    };
  }
  if (pathname === "/upload") {
    return {
      title: "Upload & Ingestion Pipeline",
      breadcrumbs: ["Platform", "Upload Documents"],
    };
  }
  if (pathname.startsWith("/review")) {
    const parts = pathname.split("/");
    const id = parts[2];
    return {
      title: "Document Review & Human Verification",
      breadcrumbs: id
        ? ["Platform", "Document Review", `${id.slice(0, 8)}...`]
        : ["Platform", "Document Review"],
    };
  }
  if (pathname === "/qa") {
    return {
      title: "Document Question Answering (O3)",
      breadcrumbs: ["Platform", "Document QA"],
    };
  }
  if (pathname === "/history") {
    return {
      title: "Document History & Repository",
      breadcrumbs: ["Platform", "Document History"],
    };
  }
  if (pathname === "/evaluation") {
    return {
      title: "Evaluation Metrics & Accuracy (O2)",
      breadcrumbs: ["Platform", "Evaluation & Analytics"],
    };
  }
  return {
    title: "Intelligent Document Processing",
    breadcrumbs: ["Platform"],
  };
}

function Layout() {
  const location = useLocation();
  const routeMeta = getRouteMetadata(location.pathname);

  const [collapsed, setCollapsed] = useState(() => {
    const saved = localStorage.getItem("idp_sidebar_collapsed");
    return saved === "true";
  });
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Responsive breakpoint tracking
  useEffect(() => {
    function handleResize() {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) {
        setMobileOpen(false);
      }
    }
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Save collapsed state
  useEffect(() => {
    localStorage.setItem("idp_sidebar_collapsed", String(collapsed));
  }, [collapsed]);

  const toggleSidebar = () => {
    if (isMobile) {
      setMobileOpen((prev) => !prev);
    } else {
      setCollapsed((prev) => !prev);
    }
  };

  const sidebarWidth = isMobile ? "270px" : collapsed ? "74px" : "260px";

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "var(--bg-app)" }}>
      {/* Mobile Backdrop */}
      {isMobile && mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(15, 23, 42, 0.55)",
            backdropFilter: "blur(2px)",
            zIndex: 40,
            transition: "opacity 0.2s ease",
          }}
        />
      )}

      {/* SaaS Sidebar Navigation */}
      <aside
        style={{
          width: sidebarWidth,
          minWidth: sidebarWidth,
          background: "linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)",
          color: "var(--text-inverse)",
          display: "flex",
          flexDirection: "column",
          position: isMobile ? "fixed" : "sticky",
          top: 0,
          left: 0,
          bottom: 0,
          height: "100vh",
          zIndex: 50,
          transform: isMobile && !mobileOpen ? "translateX(-100%)" : "none",
          transition: "width 0.22s cubic-bezier(0.4, 0, 0.2, 1), transform 0.22s cubic-bezier(0.4, 0, 0.2, 1)",
          boxShadow: isMobile && mobileOpen ? "var(--shadow-lg)" : "none",
          borderRight: "1px solid rgba(255, 255, 255, 0.08)",
          overflowX: "hidden",
          overflowY: "auto",
        }}
      >
        {/* Brand / Logo Section */}
        <div
          style={{
            padding: collapsed && !isMobile ? "20px 14px" : "20px 18px",
            display: "flex",
            alignItems: "center",
            justifyContent: collapsed && !isMobile ? "center" : "space-between",
            borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
            minHeight: "68px",
            boxSizing: "border-box",
          }}
        >
          <Link
            to="/"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              textDecoration: "none",
              color: "white",
            }}
          >
            <LogoIcon size={32} />
            {(!collapsed || isMobile) && (
              <div style={{ display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "16px", fontWeight: "700", letterSpacing: "-0.2px", color: "#ffffff" }}>
                  IDP Research
                </span>
                <span style={{ fontSize: "10px", color: "#94a3b8", letterSpacing: "0.5px", textTransform: "uppercase" }}>
                  Document AI Platform
                </span>
              </div>
            )}
          </Link>
        </div>

        {/* Navigation Groups */}
        <nav style={{ padding: "16px 10px", flex: 1, display: "flex", flexDirection: "column", gap: "20px" }}>
          {NAV_ITEMS.map((group, gIdx) => (
            <div key={gIdx}>
              {(!collapsed || isMobile) && (
                <div
                  style={{
                    fontSize: "10px",
                    fontWeight: "700",
                    letterSpacing: "0.8px",
                    color: "#64748b",
                    padding: "0 10px",
                    marginBottom: "8px",
                    textTransform: "uppercase",
                  }}
                >
                  {group.category}
                </div>
              )}
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.end}
                      onClick={() => setMobileOpen(false)}
                      title={collapsed && !isMobile ? item.label : undefined}
                      style={({ isActive }) => ({
                        display: "flex",
                        alignItems: "center",
                        gap: "12px",
                        padding: collapsed && !isMobile ? "10px 0" : "10px 14px",
                        justifyContent: collapsed && !isMobile ? "center" : "flex-start",
                        borderRadius: "8px",
                        textDecoration: "none",
                        color: isActive ? "#ffffff" : "#cbd5e1",
                        background: isActive ? "#4f46e5" : "transparent",
                        fontWeight: isActive ? "600" : "500",
                        fontSize: "13.5px",
                        transition: "all 0.15s ease",
                        boxShadow: isActive ? "0 2px 8px -2px rgba(79, 70, 229, 0.4)" : "none",
                      })}
                      onMouseEnter={(e) => {
                        if (!e.currentTarget.classList.contains("active")) {
                          e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.08)";
                          e.currentTarget.style.color = "#ffffff";
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!e.currentTarget.classList.contains("active")) {
                          e.currentTarget.style.backgroundColor = "transparent";
                          e.currentTarget.style.color = "#cbd5e1";
                        }
                      }}
                    >
                      <Icon size={18} />
                      {(!collapsed || isMobile) && <span>{item.label}</span>}
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Sidebar Footer & Collapse Toggle */}
        <div
          style={{
            padding: "14px 12px",
            borderTop: "1px solid rgba(255, 255, 255, 0.08)",
            backgroundColor: "rgba(0, 0, 0, 0.15)",
          }}
        >
          {(!collapsed || isMobile) && (
            <div style={{ marginBottom: "12px", padding: "0 6px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "#38bdf8", fontWeight: "600" }}>
                <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
                Backend: 127.0.0.1:8000
              </div>
              <div style={{ fontSize: "10.5px", color: "#64748b", marginTop: "2px" }}>
                FastAPI • PyMuPDF • LayoutLMv3
              </div>
            </div>
          )}

          {!isMobile && (
            <button
              onClick={toggleSidebar}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              style={{
                width: "100%",
                padding: "8px 10px",
                display: "flex",
                alignItems: "center",
                justifyContent: collapsed ? "center" : "space-between",
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                color: "#94a3b8",
                borderRadius: "6px",
                fontSize: "12px",
                fontWeight: "500",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.12)";
                e.currentTarget.style.color = "#ffffff";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
                e.currentTarget.style.color = "#94a3b8";
              }}
            >
              {!collapsed && <span>Collapse Sidebar</span>}
              {collapsed ? <ChevronRightIcon size={16} /> : <ChevronLeftIcon size={16} />}
            </button>
          )}
        </div>
      </aside>

      {/* Main SaaS Content Area */}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
        {/* Top Navigation Bar */}
        <header
          style={{
            height: "68px",
            background: "#ffffff",
            borderBottom: "1px solid var(--border-subtle)",
            boxShadow: "var(--shadow-xs)",
            position: "sticky",
            top: 0,
            zIndex: 30,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: isMobile ? "0 14px" : "0 28px",
            boxSizing: "border-box",
          }}
        >
          {/* Left: Sidebar Toggle, Title & Breadcrumbs */}
          <div style={{ display: "flex", alignItems: "center", gap: "16px", minWidth: 0 }}>
            {isMobile && (
              <button
                onClick={toggleSidebar}
                aria-label="Toggle Navigation Menu"
                style={{
                  background: "#f1f5f9",
                  border: "1px solid #e2e8f0",
                  borderRadius: "6px",
                  padding: "8px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#1e293b",
                }}
              >
                <MenuIcon size={18} />
              </button>
            )}

            <div style={{ minWidth: 0 }}>
              {/* Breadcrumb Trail */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "11.5px",
                  color: "#64748b",
                  marginBottom: "2px",
                }}
              >
                {routeMeta.breadcrumbs.map((crumb, idx) => (
                  <span key={idx} style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                    {idx > 0 && <span style={{ color: "#cbd5e1" }}>/</span>}
                    <span style={{ fontWeight: idx === routeMeta.breadcrumbs.length - 1 ? "600" : "400", color: idx === routeMeta.breadcrumbs.length - 1 ? "#0f172a" : "#64748b" }}>
                      {crumb}
                    </span>
                  </span>
                ))}
              </div>

              {/* Dynamic Page Title */}
              <h1
                style={{
                  fontSize: "17px",
                  fontWeight: "700",
                  color: "#0f172a",
                  margin: 0,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {routeMeta.title}
              </h1>
            </div>
          </div>

          {/* Right: Model Status, API Status & Profile Area */}
          <div style={{ display: "flex", alignItems: "center", gap: isMobile ? "8px" : "14px" }}>
            {/* AI Model Ready Badge */}
            <div
              style={{
                display: isMobile ? "none" : "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "5px 12px",
                borderRadius: "20px",
                background: "#f5f3ff",
                border: "1px solid #ddd6fe",
                color: "#6d28d9",
                fontSize: "12px",
                fontWeight: "600",
              }}
            >
              <SparklesIcon size={14} color="#7c3aed" />
              <span>LayoutLMv3 (O2)</span>
            </div>

            {/* Backend Connectivity Status */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "5px 10px",
                borderRadius: "20px",
                background: "#ecfdf5",
                border: "1px solid #a7f3d0",
                color: "#166534",
                fontSize: "12px",
                fontWeight: "600",
              }}
            >
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#16a34a" }} />
              <span>Live API</span>
            </div>

            {/* Profile Area */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                paddingLeft: isMobile ? "4px" : "10px",
                borderLeft: isMobile ? "none" : "1px solid #e2e8f0",
              }}
            >
              <div
                style={{
                  width: "36px",
                  height: "36px",
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #4f46e5 0%, #1e1b4b 100%)",
                  color: "#ffffff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: "700",
                  fontSize: "13px",
                  boxShadow: "0 2px 4px rgba(79, 70, 229, 0.25)",
                }}
              >
                RA
              </div>
              <div style={{ display: isMobile ? "none" : "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "13px", fontWeight: "600", color: "#0f172a", lineHeight: 1.2 }}>
                  Research Team
                </span>
                <span style={{ fontSize: "11px", color: "#64748b", lineHeight: 1.2 }}>
                  Admin Analyst
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Content Outlet Container */}
        <main style={{ flex: 1, padding: isMobile ? "20px 14px" : "28px", maxWidth: "1600px", width: "100%", margin: "0 auto", boxSizing: "border-box" }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default Layout;
