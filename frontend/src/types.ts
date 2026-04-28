export type Difficulty = "basic" | "moderate" | "advanced";

export type RetrievalPayload = {
  semantic_docs?: string[];
  semantic_distances?: number[];
  graph_path?: string[];
  graph_docs?: string[];
  combined_docs?: string[];
  num_semantic?: number;
  num_graph?: number;
  total_unique?: number;
};

export type RetrievalNote = {
  title: string;
  preview: string;
  full_text: string;
  relevance_score?: number | null;
  concepts: string[];
};

export type QuizItem = {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
  difficulty?: Difficulty | string;
};

export type Flashcard = {
  front: string;
  back: string;
  hint?: string;
};

export type AGCTResponse = {
  status: "success" | "rejected" | string;
  error?: string;
  query: string;
  mode: "answer" | "quiz" | "flashcards" | string;
  topic_query: string;
  used_memory_topic?: boolean;
  latest_topic?: string | null;
  difficulty: Difficulty | string;
  graph_path: string[];
  explanation: string;
  verified: boolean;
  verification_score: number;
  sources: string[];
  retrieval: RetrievalPayload;
  retrieval_notes?: RetrievalNote[];
  suggested_questions?: string[];
  chat_history?: { user: string; assistant: string; topic: string }[];
  quiz: QuizItem[];
  flashcards: Flashcard[];
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: AGCTResponse;
};

export type ConceptGraph = {
  nodes: { id: string; label: string }[];
  edges: { source: string; target: string; relation: string }[];
};

export type ProgressPayload = {
  user_id: string;
  difficulty: Difficulty | string;
  recent_topics: string[];
  progress: number;
  recommendations: string[];
};
