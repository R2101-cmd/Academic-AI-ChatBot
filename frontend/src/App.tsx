import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  BookOpen,
  BrainCircuit,
  CheckCircle,
  CheckCircle2,
  ChevronDown,
  Copy,
  FileText,
  GraduationCap,
  Layers3,
  Lightbulb,
  Menu,
  MessageSquarePlus,
  Moon,
  NotebookText,
  PlayCircle,
  RotateCcw,
  Search,
  Settings,
  Sparkles,
  Sun,
  Target,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  TrendingUp,
  Trophy,
  X,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip } from "recharts";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { api } from "./lib/api";
import { cn, compactText, scoreToPercent } from "./lib/utils";
import type { AGCTResponse, Flashcard, Message, ProgressPayload, QuizItem, RetrievalNote } from "./types";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";

const starterPrompts = [
  { icon: BookOpen, title: "Explain Topic", prompt: "Explain backpropagation step by step" },
  { icon: BrainCircuit, title: "Quiz Me", prompt: "Create a quiz on chain rule for neural networks" },
  { icon: NotebookText, title: "Flashcards", prompt: "Generate flashcards for gradient descent" },
  { icon: RotateCcw, title: "Revise Concepts", prompt: "Help me revise derivatives and the chain rule" },
];

const sampleHistory = ["Backpropagation revision", "AI ethics overview", "Calculus prerequisites"];
const favoriteTopics = ["Neural Networks", "Gradient Descent", "Calculus"];

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [progress, setProgress] = useState<ProgressPayload | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const latestResponse = [...messages].reverse().find((message) => message.response)?.response;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    api.progress().then(setProgress).catch(() => setProgress(null));
  }, [messages.length]);

  async function submit(prompt?: string) {
    const text = (prompt ?? query).trim();
    if (!text || loading) return;

    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((current) => [...current, userMessage]);
    setQuery("");
    setLoading(true);

    try {
      const response = await api.query(text);
      if (response.status !== "success") throw new Error(response.error ?? "The tutor rejected the request.");
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.explanation,
          response,
        },
      ]);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown error";
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `I could not reach the study assistant API. Start the backend with \`uvicorn backend.api:app --reload\` and try again.\n\n${detail}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 text-slate-950 dark:bg-navy-950 dark:text-slate-50">
      <Sidebar
        onNewChat={() => setMessages([])}
        onPrompt={submit}
        dark={dark}
        setDark={setDark}
        messages={messages}
        progress={progress}
        latestResponse={latestResponse}
      />

      <AnimatePresence>
        {mobileSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-navy-950/70 lg:hidden"
          >
            <motion.div initial={{ x: -320 }} animate={{ x: 0 }} exit={{ x: -320 }} className="h-full w-80 max-w-[88vw]">
              <Sidebar
                mobile
                onNewChat={() => {
                  setMessages([]);
                  setMobileSidebarOpen(false);
                }}
                onPrompt={(prompt) => {
                  setMobileSidebarOpen(false);
                  submit(prompt);
                }}
                dark={dark}
                setDark={setDark}
                messages={messages}
                progress={progress}
                latestResponse={latestResponse}
              />
            </motion.div>
            <button
              onClick={() => setMobileSidebarOpen(false)}
              title="Close menu"
              className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-lg bg-white text-navy-950"
            >
              <X size={18} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white px-5 dark:border-slate-800 dark:bg-navy-900">
          <div className="flex min-w-0 items-center gap-3">
            <Button variant="outline" size="icon" className="lg:hidden" onClick={() => setMobileSidebarOpen(true)} title="Open menu">
              <Menu size={18} />
            </Button>
            <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">
              Personal Study Companion
            </p>
            <h1 className="text-lg font-semibold">AI Study Assistant</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {latestResponse?.verified && (
              <span className="inline-flex items-center gap-2 rounded-full bg-cyan-50 px-3 py-1 text-xs font-semibold text-navy-800 dark:bg-cyanSoft/10 dark:text-cyan-100">
                <CheckCircle2 size={14} /> Reviewed Answer
              </span>
            )}
            <Button variant="outline" size="icon" onClick={() => setMessages([])} title="Clear chat">
              <Trash2 size={16} />
            </Button>
          </div>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_390px]">
          <section className="flex min-h-0 flex-col">
            <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-6 md:px-8">
              {messages.length === 0 ? (
                <WelcomeState onPrompt={submit} progress={progress} latestResponse={latestResponse} />
              ) : (
                <div className="mx-auto max-w-4xl space-y-5">
                  <AnimatePresence initial={false}>
                    {messages.map((message) => (
                      <ChatMessage key={message.id} message={message} onPrompt={submit} />
                    ))}
                  </AnimatePresence>
                  {loading && <TypingIndicator />}
                </div>
              )}
            </div>

            <form onSubmit={onSubmit} className="border-t border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-navy-900">
              <div className="mx-auto flex max-w-4xl items-end gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-2 shadow-panel dark:border-slate-800 dark:bg-navy-950">
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  rows={1}
                  placeholder="Ask me to explain a topic, create a quiz, generate flashcards, revise concepts, or help you study..."
                  className="max-h-36 min-h-11 flex-1 resize-none bg-transparent px-3 py-3 text-sm text-slate-950 outline-none placeholder:text-slate-500 dark:text-slate-100 dark:placeholder:text-slate-400"
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      submit();
                    }
                  }}
                />
                <Button disabled={!query.trim() || loading} className="mb-1">
                  <Sparkles size={16} /> Send
                </Button>
              </div>
            </form>
          </section>

          <RightPanel response={latestResponse} />
        </div>
      </main>
    </div>
  );
}

function Sidebar({
  onNewChat,
  onPrompt,
  dark,
  setDark,
  messages,
  progress,
  latestResponse,
  mobile = false,
}: {
  onNewChat: () => void;
  onPrompt: (prompt: string) => void;
  dark: boolean;
  setDark: (value: boolean) => void;
  messages: Message[];
  progress: ProgressPayload | null;
  latestResponse?: AGCTResponse;
  mobile?: boolean;
}) {
  const history = messages.filter((message) => message.role === "user").map((message) => message.content);
  const recentTopics = progress?.recent_topics?.length ? progress.recent_topics : sampleHistory;
  const recommendations = progress?.recommendations?.length
    ? progress.recommendations
    : ["Review one weak area today.", "Try a short quiz after each topic.", "Turn explanations into flashcards."];

  return (
    <aside className={cn("h-full w-72 shrink-0 flex-col border-r border-slate-200 bg-navy-950 text-white", mobile ? "flex" : "hidden lg:flex")}>
      <div className="flex h-16 items-center gap-3 border-b border-white/10 px-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-navy-950">
          <GraduationCap size={21} />
        </div>
        <div>
          <p className="text-sm font-semibold">Academic AI</p>
          <p className="text-xs text-slate-300">Study Assistant</p>
        </div>
      </div>

      <div className="space-y-4 p-4">
        <Button onClick={onNewChat} className="w-full justify-start bg-white text-navy-950 hover:bg-slate-100">
          <MessageSquarePlus size={16} /> New Chat
        </Button>

        <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2">
          <Search size={15} className="text-slate-400" />
          <input className="w-full bg-transparent text-sm outline-none placeholder:text-slate-400" placeholder="Search chats" />
        </div>
      </div>

      <nav className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 pb-4">
        <SidebarGroup title="Recently Studied" icon={<FileText size={15} />}>
          {[...history, ...sampleHistory].slice(0, 6).map((item) => (
            <button key={item} onClick={() => onPrompt(item)} className="block w-full truncate rounded-lg px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/10">
              {item}
            </button>
          ))}
        </SidebarGroup>

        <SidebarGroup title="Recommended Revisions" icon={<RotateCcw size={15} />}>
          {recentTopics.slice(0, 4).map((topic) => (
            <SidebarItem key={topic} icon={<Target size={15} />} label={topic} onClick={() => onPrompt(`Revise ${topic}`)} />
          ))}
        </SidebarGroup>

        <SidebarGroup title="Favorite Topics" icon={<BookOpen size={15} />}>
          {favoriteTopics.map((topic) => (
            <SidebarItem key={topic} icon={<CheckCircle size={15} />} label={topic} onClick={() => onPrompt(`Explain ${topic}`)} />
          ))}
        </SidebarGroup>

        <SidebarGroup title="AI Suggestions" icon={<Lightbulb size={15} />}>
          {recommendations.slice(0, 3).map((item) => (
            <button key={item} onClick={() => onPrompt(item)} className="block w-full rounded-lg px-3 py-2 text-left text-sm leading-5 text-slate-200 hover:bg-white/10">
              {item}
            </button>
          ))}
        </SidebarGroup>

        <SidebarGroup title="Study assets" icon={<NotebookText size={15} />}>
          <SidebarItem icon={<Layers3 size={15} />} label="Saved quizzes" />
          <SidebarItem icon={<BrainCircuit size={15} />} label="Saved flashcards" />
          <SidebarItem icon={<FileText size={15} />} label="Revision notes" />
        </SidebarGroup>

        <SidebarGroup title="Today" icon={<Trophy size={15} />}>
          <div className="rounded-lg bg-white/5 px-3 py-2 text-sm text-slate-200">
            <p className="font-semibold">{latestResponse?.topic_query || recentTopics[0]}</p>
            <p className="mt-1 text-xs leading-5 text-slate-400">Practice for 10 minutes, then answer a quiz.</p>
          </div>
        </SidebarGroup>
      </nav>

      <div className="border-t border-white/10 p-4">
        <Button variant="ghost" className="w-full justify-start text-slate-100 hover:bg-white/10" onClick={() => setDark(!dark)}>
          {dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? "Light mode" : "Dark mode"}
        </Button>
        <Button variant="ghost" className="mt-2 w-full justify-start text-slate-100 hover:bg-white/10">
          <Settings size={16} /> Settings
        </Button>
      </div>
    </aside>
  );
}

function SidebarGroup({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 px-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
        {icon}
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function SidebarItem({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick?: () => void }) {
  return (
    <button onClick={onClick} className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/10">
      {icon}
      {label}
    </button>
  );
}

function WelcomeState({
  onPrompt,
  progress,
  latestResponse,
}: {
  onPrompt: (prompt: string) => void;
  progress: ProgressPayload | null;
  latestResponse?: AGCTResponse;
}) {
  return (
    <div className="mx-auto max-w-5xl py-6">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600 shadow-sm dark:border-slate-800 dark:bg-navy-900 dark:text-slate-300">
          <Sparkles size={14} /> Adaptive Study Support
        </div>
        <h2 className="max-w-3xl text-3xl font-semibold tracking-tight md:text-5xl">Learn with a tutor that explains, practices, and adapts to you.</h2>
        <p className="mt-4 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-300">
          Continue a topic, practice weak areas, or turn any explanation into quiz questions and flashcards.
        </p>
      </motion.div>
      <StudyDashboard progress={progress} latestResponse={latestResponse} onPrompt={onPrompt} />
      <div className="mt-5 grid gap-3 md:grid-cols-4">
        {starterPrompts.map(({ icon: Icon, title, prompt }) => (
          <button key={title} onClick={() => onPrompt(prompt)} className="rounded-lg border border-slate-200 bg-white p-4 text-left text-sm font-medium shadow-panel transition hover:-translate-y-0.5 hover:shadow-academic dark:border-slate-800 dark:bg-navy-900">
            <span className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-navy-900 dark:bg-navy-800 dark:text-cyan-100">
              <Icon size={18} />
            </span>
            <span className="block font-semibold">{title}</span>
            <span className="mt-1 block text-xs leading-5 text-slate-500 dark:text-slate-400">{prompt}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function StudyDashboard({
  progress,
  latestResponse,
  onPrompt,
}: {
  progress: ProgressPayload | null;
  latestResponse?: AGCTResponse;
  onPrompt: (prompt: string) => void;
}) {
  const recent = progress?.recent_topics?.length ? progress.recent_topics : ["Calculus", "Chain Rule", "Backpropagation"];
  const weakArea = recent[1] ?? "Chain Rule";
  const topic = latestResponse?.topic_query || recent[0];
  const cards = [
    { icon: PlayCircle, label: "Continue Learning", value: topic, action: `Continue ${topic}` },
    { icon: TrendingUp, label: "Recent Quiz Scores", value: `${progress?.progress ?? 68}% progress`, action: `Create quiz on ${topic}` },
    { icon: AlertCircle, label: "Weak Area", value: weakArea, action: `Explain ${weakArea} simpler` },
    { icon: RotateCcw, label: "Daily Revision", value: "10-minute recap", action: `Revise ${topic}` },
  ];

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {cards.map(({ icon: Icon, label, value, action }) => (
        <button key={label} onClick={() => onPrompt(action)} className="rounded-lg border border-slate-200 bg-white p-4 text-left shadow-panel transition hover:-translate-y-0.5 hover:border-navy-700 dark:border-slate-800 dark:bg-navy-900 dark:hover:border-cyanSoft">
          <div className="flex items-center justify-between gap-3">
            <Icon size={18} className="text-navy-800 dark:text-cyan-100" />
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</span>
          </div>
          <p className="mt-3 text-sm font-semibold text-slate-900 dark:text-slate-100">{value}</p>
        </button>
      ))}
    </div>
  );
}

function ChatMessage({ message, onPrompt }: { message: Message; onPrompt: (prompt: string) => void }) {
  const isUser = message.role === "user";

  return (
    <motion.article
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("flex gap-3", isUser && "justify-end")}
    >
      {!isUser && <Avatar icon={<BrainCircuit size={18} />} />}
      <div className={cn("max-w-[88%] rounded-2xl px-4 py-3 shadow-sm", isUser ? "bg-navy-900 text-white dark:bg-slate-100 dark:text-navy-950" : "bg-white text-slate-900 dark:bg-navy-900 dark:text-slate-100")}>
        {!isUser && message.response && <ResponseMeta response={message.response} />}
        <div
          className={cn(
            "prose max-w-none text-sm",
            isUser
              ? "prose-invert dark:prose-slate"
              : "prose-slate dark:prose-invert",
          )}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
            {message.content}
          </ReactMarkdown>
        </div>
        {!isUser && (
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(message.content)}>
              <Copy size={14} /> Copy
            </Button>
          </div>
        )}
        {message.response?.quiz?.length ? <QuizBlock quiz={message.response.quiz} /> : null}
        {message.response?.flashcards?.length ? <Flashcards cards={message.response.flashcards} /> : null}
        {!isUser && message.response?.suggested_questions?.length ? (
          <SuggestionChips suggestions={message.response.suggested_questions} onPrompt={onPrompt} />
        ) : null}
        {!isUser && <ResponseFeedback />}
      </div>
      {isUser && <Avatar icon={<GraduationCap size={18} />} />}
    </motion.article>
  );
}

function ResponseFeedback() {
  const [choice, setChoice] = useState<"up" | "down" | null>(null);
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
      <span>Was this explanation helpful?</span>
      <button onClick={() => setChoice("up")} className={cn("rounded-full border px-2 py-1 transition", choice === "up" ? "border-emerald-500 bg-emerald-50 text-emerald-700" : "border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-navy-950")} title="Helpful">
        <ThumbsUp size={14} />
      </button>
      <button onClick={() => setChoice("down")} className={cn("rounded-full border px-2 py-1 transition", choice === "down" ? "border-rose-500 bg-rose-50 text-rose-700" : "border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-navy-950")} title="Not helpful">
        <ThumbsDown size={14} />
      </button>
    </div>
  );
}

function SuggestionChips({ suggestions, onPrompt }: { suggestions: string[]; onPrompt: (prompt: string) => void }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-100 pt-3 dark:border-slate-800">
      {suggestions.map((suggestion) => (
        <button
          key={suggestion}
          onClick={() => onPrompt(suggestion)}
          className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:border-navy-700 hover:bg-white dark:border-slate-700 dark:bg-navy-950 dark:text-slate-200 dark:hover:border-cyanSoft"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}

function Avatar({ icon }: { icon: React.ReactNode }) {
  return <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-slate-100 text-navy-900 dark:bg-navy-800 dark:text-slate-100">{icon}</div>;
}

function ResponseMeta({ response }: { response: AGCTResponse }) {
  const quality = response.verification_score ? `${scoreToPercent(response.verification_score)}% topic match` : "Topic match building";
  return (
    <div className="mb-3 flex flex-wrap gap-2">
      <Badge>Difficulty: {response.difficulty}</Badge>
      <Badge>{response.verified ? "Topic accuracy checked" : "Needs another example"}</Badge>
      <Badge>{quality}</Badge>
      <Badge>{response.mode === "answer" ? "Explanation" : response.mode}</Badge>
    </div>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-semibold text-slate-600 dark:border-slate-700 dark:bg-navy-950 dark:text-slate-300">{children}</span>;
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-500">
      <Avatar icon={<BrainCircuit size={18} />} />
      <div className="rounded-2xl bg-white px-4 py-3 shadow-sm dark:bg-navy-900">
        <span className="animate-pulse">Your study assistant is finding notes, checking the topic, and preparing a clear answer...</span>
      </div>
    </div>
  );
}

function QuizBlock({ quiz }: { quiz: QuizItem[] }) {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const score = useMemo(() => quiz.reduce((sum, item, index) => sum + (answers[index] === item.correct_index ? 1 : 0), 0), [answers, quiz]);

  return (
    <Card className="mt-4 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Generated Quiz</h3>
        {submitted && <Badge>Score {score}/{quiz.length}</Badge>}
      </div>
      <div className="space-y-4">
        {quiz.map((item, index) => (
          <div key={item.question} className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
            <p className="mb-2 text-sm font-medium">{index + 1}. {item.question}</p>
            <div className="grid gap-2 md:grid-cols-2">
              {item.options.map((option, optionIndex) => (
                <button
                  key={option}
                  onClick={() => setAnswers((current) => ({ ...current, [index]: optionIndex }))}
                  className={cn(
                    "rounded-lg border px-3 py-2 text-left text-sm transition",
                    answers[index] === optionIndex ? "border-navy-800 bg-navy-900 text-white" : "border-slate-200 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-navy-800",
                    submitted && optionIndex === item.correct_index && "border-emerald-500 bg-emerald-50 text-emerald-900",
                  )}
                >
                  {option}
                </button>
              ))}
            </div>
            {submitted && <p className="mt-2 text-xs text-slate-600 dark:text-slate-300">{item.explanation}</p>}
          </div>
        ))}
      </div>
      <Button className="mt-4" onClick={() => setSubmitted(true)}>Submit Quiz</Button>
    </Card>
  );
}

function Flashcards({ cards }: { cards: Flashcard[] }) {
  const [index, setIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const card = cards[index];

  return (
    <Card className="mt-4 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Flashcards</h3>
        <Badge>{index + 1}/{cards.length}</Badge>
      </div>
      <button onClick={() => setFlipped(!flipped)} className="min-h-36 w-full rounded-xl border border-slate-200 bg-slate-50 p-5 text-left transition dark:border-slate-800 dark:bg-navy-950">
        <motion.div key={`${index}-${flipped}`} initial={{ rotateX: -12, opacity: 0 }} animate={{ rotateX: 0, opacity: 1 }}>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{flipped ? "Definition" : "Concept"}</p>
          <p className="mt-3 text-lg font-semibold">{flipped ? card.back : card.front}</p>
          {!flipped && card.hint && <p className="mt-3 text-sm text-slate-500">Hint: {card.hint}</p>}
        </motion.div>
      </button>
      <div className="mt-3 flex justify-between">
        <Button variant="outline" onClick={() => { setIndex(Math.max(0, index - 1)); setFlipped(false); }}>Previous</Button>
        <Button variant="outline" onClick={() => { setIndex(Math.min(cards.length - 1, index + 1)); setFlipped(false); }}>Next</Button>
      </div>
    </Card>
  );
}

function RightPanel({ response }: { response?: AGCTResponse }) {
  return (
    <aside className="hidden min-h-0 overflow-y-auto border-l border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-navy-900 xl:block">
      <div className="space-y-4">
        <LearningPath response={response} />
        <PerformanceChart response={response} />
        <RetrievalPanel response={response} />
        <ReasoningPanel response={response} />
      </div>
    </aside>
  );
}

function PerformanceChart({ response }: { response?: AGCTResponse }) {
  const score = response?.verification_score ? Math.round(response.verification_score * 100) : 68;
  const data = [
    { label: "NLP", value: 82 },
    { label: "RAG", value: Math.max(55, score - 8) },
    { label: "CoT", value: Math.max(60, score - 3) },
    { label: "Verify", value: score },
  ];

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Pipeline Confidence</h2>
        <Badge>{score}%</Badge>
      </div>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d7dde6" />
            <Tooltip />
            <Area type="monotone" dataKey="value" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.18} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function LearningPath({ response }: { response?: AGCTResponse }) {
  const path = response?.graph_path?.length
    ? response.graph_path
    : ["Calculus", "Derivatives", "Chain Rule", "Backpropagation"];

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Concept Progression</h2>
        <Badge>{path.length} concepts</Badge>
      </div>
      <div className="space-y-1">
        {path.map((node, index) => (
          <LearningPathNode key={`${node}-${index}`} node={node} index={index} total={path.length} previous={path[index - 1]} />
        ))}
      </div>
    </Card>
  );
}

function LearningPathNode({ node, index, total, previous }: { node: string; index: number; total: number; previous?: string }) {
  const [open, setOpen] = useState(false);
  const complete = index < Math.max(1, Math.ceil(total * 0.55));
  return (
    <div className="relative pl-4">
      {index > 0 && <div className="absolute left-[21px] top-0 h-4 w-px bg-slate-300 dark:bg-slate-700" />}
      <button
        onClick={() => setOpen(!open)}
        title={previous ? `Builds on ${previous}` : "Starting concept"}
        className="flex w-full items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-left transition hover:border-navy-700 hover:bg-white dark:border-slate-800 dark:bg-navy-950 dark:hover:border-cyanSoft"
      >
        <span className={cn("mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-xs font-semibold", complete ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-400/15 dark:text-emerald-100" : "bg-white text-navy-900 dark:bg-navy-800 dark:text-slate-100")}>
          {complete ? <CheckCircle size={16} /> : index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-slate-800 dark:text-slate-100">{node}</span>
          <span className="mt-1 block text-xs leading-5 text-slate-500 dark:text-slate-400">
            {previous ? `Prerequisite: ${previous}` : "Start here before advanced ideas."}
          </span>
          {open && (
            <span className="mt-2 block rounded-md bg-white px-3 py-2 text-xs leading-5 text-slate-600 dark:bg-navy-900 dark:text-slate-300">
              Master this concept by reviewing its definition, one worked example, and how it supports the next step.
            </span>
          )}
        </span>
        <ChevronDown size={15} className={cn("mt-1 transition", open && "rotate-180")} />
      </button>
      {index < total - 1 && <div className="ml-[17px] flex h-5 items-center text-slate-400"><ChevronDown size={16} /></div>}
    </div>
  );
}

function RetrievalPanel({ response }: { response?: AGCTResponse }) {
  const notes = response?.retrieval_notes ?? [];
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Study Notes Found</h2>
        <Badge>{notes.length || response?.retrieval?.total_unique || 0} notes</Badge>
      </div>
      <div className="space-y-2">
        {notes.length ? (
          notes.slice(0, 4).map((note, index) => <RetrievalNoteCard key={`${note.title}-${index}`} note={note} />)
        ) : (
          <p className="rounded-lg border border-slate-200 p-3 text-xs leading-5 text-slate-600 dark:border-slate-800 dark:text-slate-300">
            Helpful study notes will appear here after a question, with relevance, previews, and key concepts.
          </p>
        )}
      </div>
    </Card>
  );
}

function RetrievalNoteCard({ note }: { note: RetrievalNote }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{note.title}</p>
          {note.concepts.length ? (
            <div className="mt-1 flex flex-wrap gap-1">
              {note.concepts.map((concept) => (
                <span key={concept} className="rounded-full bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-navy-800 dark:bg-cyanSoft/10 dark:text-cyan-100">
                  {concept}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        {note.relevance_score !== null && note.relevance_score !== undefined ? (
          <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-600 dark:bg-navy-950 dark:text-slate-300">
            {Math.round(note.relevance_score * 100)}% Relevant
          </span>
        ) : null}
      </div>
      <p className="text-xs leading-5 text-slate-600 dark:text-slate-300">
        {open ? note.full_text : compactText(note.preview, 240)}
      </p>
      {note.full_text.length > note.preview.length ? (
        <button onClick={() => setOpen(!open)} className="mt-2 text-xs font-semibold text-navy-800 dark:text-cyan-100">
          {open ? "View less" : "View more"}
        </button>
      ) : null}
    </div>
  );
}

function ReasoningPanel({ response }: { response?: AGCTResponse }) {
  const quality = response?.verification_score ? `${scoreToPercent(response.verification_score)}%` : "Ready";
  const accuracy = response ? (response.verified ? "Checked" : "Review suggested") : "Waiting";
  return (
    <Card className="p-4">
      <h2 className="mb-3 text-sm font-semibold">Study Guidance</h2>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between rounded-lg bg-slate-50 p-2 dark:bg-navy-950">
          <span>Difficulty Level</span>
          <strong className="capitalize">{response?.difficulty ?? "moderate"}</strong>
        </div>
        <div className="flex justify-between rounded-lg bg-slate-50 p-2 dark:bg-navy-950">
          <span>Topic Accuracy</span>
          <strong>{accuracy}</strong>
        </div>
        <div className="flex justify-between rounded-lg bg-slate-50 p-2 dark:bg-navy-950">
          <span>Verification</span>
          <strong>{response?.verification_status ?? "Ready"}</strong>
        </div>
        <div className="flex justify-between rounded-lg bg-slate-50 p-2 dark:bg-navy-950">
          <span>Explanation Quality</span>
          <strong>{quality}</strong>
        </div>
      </div>
    </Card>
  );
}
