import type { AGCTResponse, ConceptGraph, ProgressPayload } from "../types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  query(query: string, userId = "default") {
    return request<AGCTResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify({ query, user_id: userId }),
    });
  },
  graph() {
    return request<ConceptGraph>("/api/graph");
  },
  progress(userId = "default") {
    return request<ProgressPayload>(`/api/progress/${userId}`);
  },
  submitQuizScore(score: number, userId = "default") {
    return request<{ status: string; difficulty: string }>("/api/quiz-score", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, score }),
    });
  },
};
