import React, { useState, useMemo } from 'react';
import { 
  Search, 
  Terminal, 
  ExternalLink, 
  Github, 
  Zap, 
  BookOpen, 
  Layers, 
  Cpu, 
  ShieldCheck, 
  MessageSquare,
  TrendingUp,
  Map,
  Code,
  CheckCircle2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import resourcesData from './data/resources.json';

const IconMap = {
  "Learning & Education": <BookOpen className="w-5 h-5" />,
  "Frameworks & Libraries": <Layers className="w-5 h-5" />,
  "MLOps & Deployment": <Terminal className="w-5 h-5" />,
  "Datasets": <Cpu className="w-5 h-5" />,
  "AI Tools & Apps": <Zap className="w-5 h-5" />,
  "Ethics, Safety & Governance": <ShieldCheck className="w-5 h-5" />,
  "Communities & Events": <MessageSquare className="w-5 h-5" />,
};

const ResourceCard = ({ item }) => (
  <motion.div 
    layout
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, scale: 0.95 }}
    className="glass p-5 rounded-2xl border border-white/5 hover:border-primary/50 transition-all group flex flex-col h-full"
  >
    <div className="flex justify-between items-start mb-3">
      <span className={`difficulty-badge ${item.difficulty === 'Beginner' ? 'difficulty-beginner' : item.difficulty === 'Intermediate' ? 'difficulty-intermediate' : 'difficulty-advanced'}`}>
        {item.difficulty}
      </span>
      <a href={item.link} target="_blank" rel="noopener noreferrer" className="text-muted hover:text-primary transition-colors">
        <ExternalLink className="w-4 h-4" />
      </a>
    </div>
    <h4 className="text-lg font-bold mb-2 group-hover:text-primary transition-colors">{item.name}</h4>
    <p className="text-muted text-sm flex-grow">{item.description || "Explore this powerful AI resource."}</p>
    <div className="mt-4 pt-4 border-t border-white/5">
      <a 
        href={item.link} 
        target="_blank" 
        rel="noopener noreferrer"
        className="text-xs font-semibold text-primary uppercase tracking-wider flex items-center gap-2 hover:gap-3 transition-all"
      >
        View Resource <ExternalLink className="w-3 h-3" />
      </a>
    </div>
  </motion.div>
);

const SectionHeader = ({ title, icon }) => (
  <div className="flex items-center gap-3 mb-8 mt-12">
    <div className="p-2 rounded-lg bg-primary/10 text-primary">
      {icon}
    </div>
    <h2 className="text-3xl font-bold">{title}</h2>
  </div>
);

export default function App() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const filteredData = useMemo(() => {
    let result = resourcesData.categories;
    
    if (activeCategory !== "All") {
      result = result.filter(cat => cat.name === activeCategory);
    }

    if (searchQuery) {
      return result.map(cat => ({
        ...cat,
        subcategories: cat.subcategories.map(sub => ({
          ...sub,
          items: sub.items.filter(item => 
            item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            item.description.toLowerCase().includes(searchQuery.toLowerCase())
          )
        })).filter(sub => sub.items.length > 0)
      })).filter(cat => cat.subcategories.length > 0);
    }

    return result;
  }, [searchQuery, activeCategory]);

  return (
    <div className="min-h-screen pb-20">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 glass border-b border-white/5 py-4">
        <div className="container flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Zap className="text-white w-5 h-5 fill-current" />
            </div>
            <span className="font-extrabold text-xl tracking-tight hidden md:block">Ultimate AI</span>
          </div>
          <div className="relative max-w-md w-full mx-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted w-4 h-4" />
            <input 
              type="text" 
              placeholder="Search resources..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-full py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-primary/50 transition-all"
            />
          </div>
          <div className="flex items-center gap-4">
            <a href="https://github.com/DhanushNehru/Ultimate-AI-Resources" target="_blank" rel="noopener noreferrer" className="text-muted hover:text-white transition-colors">
              <Github className="w-6 h-6" />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="container pt-20 pb-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <span className="px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-widest mb-6 inline-block border border-primary/20">
            Curated by Experts
          </span>
          <h1 className="text-5xl md:text-7xl font-black mb-6 leading-tight">
            The Hub of <span className="gradient-text">Artificial Intelligence</span>
          </h1>
          <p className="text-muted text-lg md:text-xl max-w-2xl mx-auto mb-10">
            A meticulously curated collection of AI tools, frameworks, datasets, and roadmaps to take you from beginner to expert.
          </p>
        </motion.div>
      </header>

      {/* Filter Tabs */}
      <div className="container mb-12 flex flex-wrap gap-2 justify-center">
        {["All", ...resourcesData.categories.map(c => c.name)].map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-5 py-2 rounded-full text-sm font-semibold transition-all ${
              activeCategory === cat 
                ? 'bg-primary text-white shadow-lg shadow-primary/25' 
                : 'bg-white/5 text-muted hover:bg-white/10'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <main className="container">
        <AnimatePresence mode="popLayout">
          {filteredData.map((category) => (
            <motion.section 
              key={category.name}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-16"
            >
              <SectionHeader title={category.name} icon={IconMap[category.name] || <Code />} />
              
              <div className="space-y-12">
                {category.subcategories.map((sub) => (
                  <div key={sub.name}>
                    <h3 className="text-xl font-bold mb-6 text-muted flex items-center gap-2">
                      <span className="w-1 h-1 rounded-full bg-primary" />
                      {sub.name}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {sub.items.map((item, idx) => (
                        <ResourceCard key={`${item.name}-${idx}`} item={item} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </motion.section>
          ))}
        </AnimatePresence>

        {filteredData.length === 0 && (
          <div className="text-center py-20">
            <Search className="w-12 h-12 text-muted mx-auto mb-4 opacity-20" />
            <h3 className="text-xl font-bold text-muted">No resources found matching your search.</h3>
          </div>
        )}
      </main>

      {/* Trending Sidebar / Bottom Section */}
      <section className="container mt-20">
        <div className="glass p-8 rounded-3xl border border-primary/20 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-8 opacity-10">
            <TrendingUp className="w-32 h-32 text-primary" />
          </div>
          <div className="relative z-10">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="text-primary w-6 h-6" />
              <h2 className="text-2xl font-bold">Trending AI Tools (2025)</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {resourcesData.trending.slice(0, 8).map((tool) => (
                <a 
                  key={tool.name} 
                  href={tool.link} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="bg-white/5 p-4 rounded-xl border border-white/5 hover:bg-white/10 transition-all"
                >
                  <span className="text-[10px] font-bold text-primary uppercase tracking-tighter mb-1 block">{tool.tag}</span>
                  <h4 className="font-bold text-sm mb-1">{tool.name}</h4>
                  <p className="text-muted text-xs line-clamp-1">{tool.description}</p>
                </a>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Quick Start Section */}
      <section className="container mt-20">
        <SectionHeader title="Quick Start Guides" icon={<Zap />} />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {resourcesData.quick_start.map((guide) => (
            <div key={guide.title} className="glass rounded-2xl overflow-hidden border border-white/10">
              <div className="bg-white/5 px-6 py-4 flex justify-between items-center border-b border-white/10">
                <span className="font-bold text-sm">{guide.title}</span>
                <Code className="w-4 h-4 text-muted" />
              </div>
              <div className="p-6 overflow-x-auto">
                <pre className="text-sm font-mono text-secondary">
                  <code>{guide.code}</code>
                </pre>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Learning Roadmap */}
      <section className="container mt-20">
        <div className="text-center mb-12">
          <SectionHeader title="AI Learning Roadmap" icon={<Map />} />
          <p className="text-muted max-w-xl mx-auto -mt-4">
            Your step-by-step guide to becoming an AI Expert, from foundations to production-ready MLOps.
          </p>
        </div>
        <div className="glass p-10 rounded-3xl border border-white/5 flex flex-col md:flex-row items-center justify-between gap-12 overflow-hidden">
          <div className="space-y-6 md:w-1/2">
            {[
              { level: "Beginner", desc: "Python, Math, and ML Basics" },
              { level: "Intermediate", desc: "Deep Learning & Neural Networks" },
              { level: "Advanced", desc: "Transformers, LLMs & NLP" },
              { level: "Expert", desc: "MLOps & Scalable Production" },
            ].map((step, i) => (
              <div key={step.level} className="flex items-center gap-4 group">
                <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center font-bold text-primary group-hover:bg-primary group-hover:text-white transition-all">
                  {i + 1}
                </div>
                <div>
                  <h4 className="font-bold">{step.level}</h4>
                  <p className="text-muted text-sm">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="md:w-1/2 relative">
             <div className="absolute inset-0 bg-primary/20 blur-[100px] rounded-full" />
             <div className="relative z-10 glass p-6 rounded-2xl border border-primary/30 rotate-3">
                <p className="font-mono text-xs text-primary mb-4">// Mermaid Roadmap Preview</p>
                <div className="text-muted font-mono text-[10px] space-y-1">
                  {resourcesData.roadmap.split('\n').slice(0, 10).map((line, i) => (
                    <div key={i}>{line}</div>
                  ))}
                  <div>... and more</div>
                </div>
             </div>
          </div>
        </div>
      </section>

      <footer className="container mt-32 border-t border-white/5 pt-12 flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
            <Zap className="text-white w-4 h-4 fill-current" />
          </div>
          <span className="font-bold">Ultimate AI Resources</span>
        </div>
        <p className="text-muted text-sm">
          © 2026 Developed by Dhanush Nehru.
        </p>
        <div className="flex items-center gap-6">
          <a href="#" className="text-muted hover:text-white transition-colors text-sm">Contribute</a>
          <a href="#" className="text-muted hover:text-white transition-colors text-sm">About</a>
          <a href="https://github.com/DhanushNehru/Ultimate-AI-Resources" className="text-muted hover:text-white transition-colors">
            <Github className="w-5 h-5" />
          </a>
        </div>
      </footer>
    </div>
  );
}
