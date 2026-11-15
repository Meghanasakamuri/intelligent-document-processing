import { useEffect, useState } from "react";

function Evaluation() {
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    const history =
      JSON.parse(
        localStorage.getItem("documentHistory")
      ) || [];

    setDocuments(history);
  }, []);

  const totalDocuments = documents.length;

  const verifiedDocuments = documents.filter(
    (document) => document.verified === true
  ).length;

  const reviewDocuments =
    totalDocuments - verifiedDocuments;

  const averageScore =
    totalDocuments > 0
      ? Math.round(
          documents.reduce(
            (sum, document) =>
              sum +
              (document.invoice_fields
                ?.confidence_score || 0),
            0
          ) / totalDocuments
        )
      : 0;

  const fieldNames = [
    {
      key: "company_name",
      label: "Company Name",
    },
    {
      key: "gst_number",
      label: "GST Number",
    },
    {
      key: "invoice_number",
      label: "Invoice Number",
    },
    {
      key: "invoice_date",
      label: "Invoice Date",
    },
    {
      key: "total_amount",
      label: "Total Amount",
    },
  ];

  const getFieldStats = (field) => {
    const extracted = documents.filter(
      (document) => {
        const value =
          document.invoice_fields?.[
            field
          ];

        return (
          value &&
          value !== "Not found"
        );
      }
    ).length;

    const percentage =
      totalDocuments > 0
        ? Math.round(
            (extracted /
              totalDocuments) *
              100
          )
        : 0;

    return {
      extracted,
      percentage,
    };
  };

  return (
    <div>
      {/* Header */}

      <div style={{ marginBottom: "30px" }}>
        <h1>System Evaluation</h1>

        <p
          style={{
            color: "#64748b",
          }}
        >
          Evaluate document extraction and human verification performance.
        </p>
      </div>

      {/* Summary Cards */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "20px",
          marginBottom: "30px",
        }}
      >
        <MetricCard
          title="Documents Evaluated"
          value={totalDocuments}
        />

        <MetricCard
          title="Average Extraction Score"
          value={`${averageScore}%`}
        />

        <MetricCard
          title="Verified Documents"
          value={verifiedDocuments}
        />

        <MetricCard
          title="Needs Review"
          value={reviewDocuments}
        />
      </div>

      {/* Overall Performance */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          marginBottom: "25px",
        }}
      >
        <h2>Overall Extraction Performance</h2>

        <p
          style={{
            color: "#64748b",
          }}
        >
          Average field extraction score across all processed documents.
        </p>

        <div
          style={{
            marginTop: "25px",
            height: "18px",
            background: "#e2e8f0",
            borderRadius: "20px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${averageScore}%`,
              height: "100%",
              background: "#2563eb",
              borderRadius: "20px",
              transition: "width 0.5s ease",
            }}
          />
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "8px",
            color: "#64748b",
            fontSize: "13px",
          }}
        >
          <span>0%</span>
          <strong>
            {averageScore}%
          </strong>
          <span>100%</span>
        </div>
      </div>

      {/* Field Extraction Analysis */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
        }}
      >
        <h2>Field Extraction Analysis</h2>

        <p
          style={{
            color: "#64748b",
            marginBottom: "25px",
          }}
        >
          Percentage of documents where each target field was successfully extracted.
        </p>

        {fieldNames.map((field) => {
          const stats =
            getFieldStats(field.key);

          return (
            <div
              key={field.key}
              style={{
                marginBottom: "25px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent:
                    "space-between",
                  marginBottom: "8px",
                }}
              >
                <strong>
                  {field.label}
                </strong>

                <span
                  style={{
                    color: "#64748b",
                  }}
                >
                  {stats.extracted}/
                  {totalDocuments}{" "}
                  documents ({stats.percentage}%)
                </span>
              </div>

              <div
                style={{
                  height: "12px",
                  background: "#e2e8f0",
                  borderRadius: "20px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${stats.percentage}%`,
                    height: "100%",
                    background: "#2563eb",
                    borderRadius: "20px",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Evaluation Note */}

      <div
        style={{
          marginTop: "25px",
          padding: "20px",
          background: "#eff6ff",
          border: "1px solid #bfdbfe",
          borderRadius: "12px",
        }}
      >
        <strong>
          Prototype Evaluation
        </strong>

        <p
          style={{
            margin:
              "8px 0 0",
            color: "#475569",
            lineHeight: "1.6",
          }}
        >
          The current evaluation uses successful field extraction
          and document verification as prototype-level performance
          indicators. Model-level evaluation using benchmark datasets
          will be added in the later implementation stage.
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
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
          margin: "0 0 12px",
          color: "#64748b",
          fontSize: "14px",
        }}
      >
        {title}
      </p>

      <div
        style={{
          fontSize: "32px",
          fontWeight: "700",
          color: "#2563eb",
        }}
      >
        {value}
      </div>
    </div>
  );
}

export default Evaluation;