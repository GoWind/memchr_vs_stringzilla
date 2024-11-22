use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

const LINES_PER_DOCUMENT: usize = 1000;

#[derive(Debug)]
struct TfidfCalculator {
    document_frequency: HashMap<String, usize>,
    term_frequencies: Vec<HashMap<String, usize>>,
    n_documents: usize,
}

impl TfidfCalculator {
    fn new() -> Self {
        TfidfCalculator {
            document_frequency: HashMap::new(),
            term_frequencies: Vec::new(),
            n_documents: 0,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>())
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn process_document(&mut self, text: &str) {
        let tokens = Self::tokenize(text);
        let mut term_freq = HashMap::new();
        let mut seen_terms = HashMap::new();

        for token in tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
            seen_terms.insert(token, true);
        }

        for term in seen_terms.keys() {
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }

        self.term_frequencies.push(term_freq);
        self.n_documents += 1;
    }

    fn calculate_tfidf(&self) -> Vec<HashMap<String, f64>> {
        let mut tfidf_scores = Vec::new();

        for doc_tf in &self.term_frequencies {
            let mut doc_tfidf = HashMap::new();
            
            for (term, freq) in doc_tf {
                if let Some(&df) = self.document_frequency.get(term) {
                    let tf = *freq as f64;
                    let idf = (self.n_documents as f64 / df as f64).ln();
                    let tfidf = tf * idf;
                    doc_tfidf.insert(term.clone(), tfidf);
                }
            }
            
            tfidf_scores.push(doc_tfidf);
        }

        tfidf_scores
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut calculator = TfidfCalculator::new();

    // Process the file in chunks of LINES_PER_DOCUMENT lines
    let mut current_document = String::new();
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        current_document.push_str(&line);
        current_document.push('\n');
        line_count += 1;

        // When we reach LINES_PER_DOCUMENT lines or the end of the file,
        // process the current document
        if line_count == LINES_PER_DOCUMENT {
            calculator.process_document(&current_document);
            current_document.clear();
            line_count = 0;
        }
    }

    // Process any remaining lines as the last document
    if !current_document.is_empty() {
        calculator.process_document(&current_document);
    }

    // Calculate and print TF-IDF scores
    let tfidf_scores = calculator.calculate_tfidf();

    println!("\nTF-IDF Scores by Document:");
    for (doc_idx, doc_scores) in tfidf_scores.iter().enumerate() {
        println!("\nDocument {} (Lines {}-{})", 
                doc_idx + 1, 
                doc_idx * LINES_PER_DOCUMENT + 1, 
                (doc_idx + 1) * LINES_PER_DOCUMENT);
        
        // Sort terms by TF-IDF score
        let mut scores: Vec<_> = doc_scores.iter().collect();
        scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        // Print top 10 terms with highest TF-IDF scores
        for (term, score) in scores.iter().take(10) {
            println!("{:<20} {:.4}", term, score);
        }
    }

    // Print some statistics
    println!("\nProcessing Summary:");
    println!("Total documents processed: {}", calculator.n_documents);
    println!("Total unique terms: {}", calculator.document_frequency.len());
    println!("Lines per document: {}", LINES_PER_DOCUMENT);

    Ok(())
}
