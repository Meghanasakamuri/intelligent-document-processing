const API_BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url, options);

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail?.message || errorJson.detail || errorJson.message || response.statusText;
    } catch {
      // response wasn't JSON
    }
    throw new Error(errorDetail || "API request failed");
  }

  return response.json();
}

/**
 * Upload and process a document file (PDF or image).
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/upload", {
    method: "POST",
    body: formData,
  });
}

/**
 * Query documents with optional filters and pagination.
 */
export async function getDocuments({ q, document_type, verified, status, limit = 50, offset = 0 } = {}) {
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (document_type && document_type !== "All") params.set("document_type", document_type);
  if (verified !== undefined && verified !== null) params.set("verified", verified);
  if (status) params.set("status", status);
  if (limit) params.set("limit", limit);
  if (offset) params.set("offset", offset);

  const query = params.toString();
  return request(`/documents${query ? `?${query}` : ""}`);
}

/**
 * Get full metadata and extracted text for a specific document.
 */
export async function getDocument(id) {
  return request(`/documents/${id}`);
}

/**
 * Update document extracted fields (records an audit log entry on the backend).
 */
export async function updateDocument(id, { invoice_fields, extracted_fields }) {
  return request(`/documents/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ invoice_fields, extracted_fields }),
  });
}

/**
 * Toggle document human verification status.
 */
export async function verifyDocument(id, verified = true) {
  return request(`/documents/${id}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verified }),
  });
}

/**
 * Save verified ground truth for benchmark evaluation.
 */
export async function saveGroundTruth(id, fields) {
  return request(`/documents/${id}/ground-truth`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields }),
  });
}

/**
 * Delete a document record and its uploaded file.
 */
export async function deleteDocument(id) {
  return request(`/documents/${id}`, {
    method: "DELETE",
  });
}

/**
 * Construct URL to preview or download the original uploaded document.
 */
export function getDocumentFileUrl(id) {
  return `${API_BASE_URL}/documents/${id}/file`;
}

/**
 * Retrieve system dashboard statistics.
 */
export async function getStats() {
  return request("/stats");
}

/**
 * Retrieve evaluation metrics (Accuracy, Precision, Recall, F1 against ground truth).
 */
export async function getEvaluation() {
  return request("/evaluation");
}

/**
 * Ask a natural language question about an uploaded document (O3).
 */
export async function askQuestion(id, question) {
  return request(`/documents/${id}/qa`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

/**
 * Run advanced LayoutLMv3 layout-aware key-field extraction (O2).
 */
export async function runLayoutLMExtraction(id) {
  return request(`/documents/${id}/extract/layoutlm`, {
    method: "POST",
  });
}

/**
 * Get human-in-the-loop audit trail history for a document.
 */
export async function getAuditLog(id) {
  return request(`/documents/${id}/audit-log`);
}

/**
 * Check LayoutLMv3 and Document QA model availability status.
 */
export async function getModelStatus() {
  return request("/models/status");
}

/**
 * Pre-load deep learning models into memory.
 */
export async function loadModels(model = "all") {
  return request("/models/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
}
