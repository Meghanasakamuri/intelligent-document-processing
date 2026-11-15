import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

function Dashboard() {
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    loadHistory();

    const handleStorageChange = () => {
      loadHistory();
    };

    window.addEventListener(
      "storage",
      handleStorageChange
    );

    return () => {
      window.removeEventListener(
        "storage",
        handleStorageChange
      );
    };
  }, []);

  const loadHistory = () => {
    const history =
      JSON.parse(
        localStorage.getItem("documentHistory")
      ) || [];

    setDocuments(history);
  };

  const totalDocuments = documents.length;

  const verifiedDocuments = documents.filter(
    (document) => document.verified === true
  ).length;

  const reviewDocuments = documents.filter(
    (document) => document.verified !== true
  ).length;

  const processedDocuments = documents.filter(
    (document) => document.message
  ).length;

  return (
    <div>
      {/* Header */}

      <div style={{ marginBottom: "30px" }}>
        <h1
          style={{
            fontSize: "42px",
            marginBottom: "5px",
          }}
        >
          System Performance Dashboard
        </h1>

        <p
          style={{
            color: "#64748b",
            fontSize: "16px",
          }}
        >
          Monitor document ingestion, OCR extraction and validation.
        </p>
      </div>

      {/* Statistics */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(210px, 1fr))",
          gap: "20px",
          marginBottom: "30px",
        }}
      >
        <StatCard
          title="Documents Processed"
          value={totalDocuments}
          description="Total uploaded documents"
        />

        <StatCard
          title="Auto Processed"
          value={processedDocuments}
          description="Successfully processed"
        />

        <StatCard
          title="Needs Review"
          value={reviewDocuments}
          description="Documents awaiting verification"
        />

        <StatCard
          title="Verified"
          value={verifiedDocuments}
          description="Human verified documents"
        />
      </div>

      {/* Quick Actions */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          marginBottom: "30px",
        }}
      >
        <h2
          style={{
            textAlign: "center",
            marginBottom: "8px",
          }}
        >
          Quick Actions
        </h2>

        <p
          style={{
            textAlign: "center",
            color: "#64748b",
            marginBottom: "20px",
          }}
        >
          Start processing a document or review previously extracted data.
        </p>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "15px",
            flexWrap: "wrap",
          }}
        >
          <Link
            to="/upload"
            style={{
              padding: "13px 22px",
              background: "#2563eb",
              color: "white",
              textDecoration: "none",
              borderRadius: "8px",
              fontWeight: "600",
            }}
          >
            Upload Document
          </Link>

          <Link
            to="/review"
            style={{
              padding: "13px 22px",
              background: "#e2e8f0",
              color: "#1e293b",
              textDecoration: "none",
              borderRadius: "8px",
              fontWeight: "600",
            }}
          >
            Review Documents
          </Link>
        </div>
      </div>

      {/* Processing Pipeline */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
        }}
      >
        <h2 style={{ marginBottom: "25px" }}>
          Processing Pipeline
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "15px",
          }}
        >
          <PipelineStep
            number="01"
            title="Upload"
            description="Document ingestion"
          />

          <PipelineStep
            number="02"
            title="OCR"
            description="Text extraction"
          />

          <PipelineStep
            number="03"
            title="Extraction"
            description="Field identification"
          />

          <PipelineStep
            number="04"
            title="Review"
            description="Human validation"
          />

          <PipelineStep
            number="05"
            title="Export"
            description="Structured output"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  description,
}) {
  return (
    <div
      style={{
        background: "white",
        padding: "25px",
        borderRadius: "12px",
        border: "1px solid #e2e8f0",
        textAlign: "center",
      }}
    >
      <p
        style={{
          color: "#64748b",
          marginBottom: "12px",
          fontSize: "14px",
        }}
      >
        {title}
      </p>

      <div
        style={{
          fontSize: "34px",
          fontWeight: "700",
          color: "#2563eb",
          marginBottom: "8px",
        }}
      >
        {value}
      </div>

      <p
        style={{
          color: "#94a3b8",
          fontSize: "13px",
          lineHeight: "1.5",
          margin: 0,
        }}
      >
        {description}
      </p>
    </div>
  );
}

function PipelineStep({
  number,
  title,
  description,
}) {
  return (
    <div
      style={{
        padding: "20px",
        background: "#f8fafc",
        borderRadius: "10px",
        border: "1px solid #e2e8f0",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: "13px",
          fontWeight: "700",
          color: "#2563eb",
          marginBottom: "10px",
        }}
      >
        {number}
      </div>

      <h3
        style={{
          margin: "0 0 7px",
          fontSize: "17px",
        }}
      >
        {title}
      </h3>

      <p
        style={{
          margin: 0,
          color: "#64748b",
          fontSize: "13px",
        }}
      >
        {description}
      </p>
    </div>
  );
}

export default Dashboard;