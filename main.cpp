#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <functional>
#include <regex>

// Структура для хранения корпуса документов и метаданных TF-IDF
struct Corpus {
    // Имя файла - список слов документа
    std::unordered_map<std::string, std::vector<std::string>> docs;
    // Слово - количество документов, в которых оно встречается (df)
    std::unordered_map<std::string, int> df;
    // Общее количество документов
    int N = 0;
};

// Предобработка текста: нижний регистр + удаление пунктуации(кроме дефисов) + токенизация
std::vector<std::string> preprocess_and_tokenize(const std::string& raw_text) {
    std::string cleaned = raw_text;

    // 1. Приведение к нижнему регистру
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(),
        [](unsigned char c) { return std::tolower(c); });

    // 2. Замена комбинаций пробел-дефис-пробел на пробелы
    cleaned = std::regex_replace(cleaned, std::regex(" - "), " ");

    // 3. Удаление остальных знаков препинания
    cleaned.erase(
        std::remove_if(cleaned.begin(), cleaned.end(),
            [](unsigned char c) { return std::ispunct(c) && c != '-'; }),
        cleaned.end());

    // 4. Токенизация по пробелам
    std::istringstream iss(cleaned);
    std::vector<std::string> tokens{
        std::istream_iterator<std::string>{iss}, 
        std::istream_iterator<std::string>{}
    };

    // 5. Удаляем токены, состоящие только из дефисов (остаточные)
    tokens.erase(
        std::remove_if(tokens.begin(), tokens.end(), 
            [](const std::string& t) { 
                return t.empty() || t == "-"; 
            }),
        tokens.end());

    return tokens;
}

// Загрузка имён документов из файла
std::vector<std::string> load_document_list(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return {};
    }
    return { std::istream_iterator<std::string>(file), std::istream_iterator<std::string>() };
}

// Загрузка и препроцессинг всех документов
void load_corpus(Corpus& corpus, const std::vector<std::string>& doc_names) {
    corpus.N = static_cast<int>(doc_names.size());
    std::for_each(doc_names.begin(), doc_names.end(), [&](const std::string& fname) {
        std::ifstream ifs(fname);
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        corpus.docs[fname] = preprocess_and_tokenize(content);
    });
}

// Вычисление IDF (df) для каждого слова
void compute_df(Corpus& corpus) {
    std::for_each(corpus.docs.begin(), corpus.docs.end(), [&](const auto& pair) {
        // set автоматически удаляет дубликаты слов внутри одного документа
        std::set<std::string> unique_words(pair.second.begin(), pair.second.end());
        std::for_each(unique_words.begin(), unique_words.end(), [&](const std::string& w) {
            corpus.df[w]++;
        });
    });
}

// Вспомогательные функции расчёта метрик
double calc_tf(const std::vector<std::string>& doc_words, const std::string& word) {
    if (doc_words.empty()) return 0.0;
    int count = std::count(doc_words.begin(), doc_words.end(), word);
    return static_cast<double>(count) / doc_words.size();
}

double calc_idf(const Corpus& corpus, const std::string& word) {
    auto it = corpus.df.find(word);
    if (it == corpus.df.end()) return 0.0;
    return std::log(static_cast<double>(corpus.N) / it->second);
}

double calc_tfidf(const Corpus& corpus, const std::vector<std::string>& doc_words, const std::string& word) {
    return calc_tf(doc_words, word) * calc_idf(corpus, word);
}

// Обработчики запросов

// 1. WORD <word>
void handle_WORD(const Corpus& corpus, const std::string& word) {
    int df_count = corpus.df.count(word) ? corpus.df.at(word) : 0;
    
    std::cout << "Word: " << word << "\n"
              << "Documents total: " << corpus.N << "\n"
              << "Documents with word: " << df_count << "\n"
              << "IDF: " << std::fixed << std::setprecision(4) << calc_idf(corpus, word) << "\n"
              << "Appears in:\n";

    if (df_count > 0) {
        std::vector<std::string> appears_in;
        // Находим документы, содержащие слово
        std::for_each(corpus.docs.begin(), corpus.docs.end(), [&](const auto& pair) {
            if (std::find(pair.second.begin(), pair.second.end(), word) != pair.second.end()) {
                appears_in.push_back(pair.first);
            }
        });
        std::sort(appears_in.begin(), appears_in.end());
        std::for_each(appears_in.begin(), appears_in.end(), [](const std::string& f) {
            std::cout << "- " << f << "\n";
        });
    }
}

// 2. WORD_IN_DOC <word> <document>
void handle_WORD_IN_DOC(const Corpus& corpus, const std::string& word, const std::string& doc) {
    auto it = corpus.docs.find(doc);
    if (it == corpus.docs.end()) {
        std::cerr << "Document not found: " << doc << "\n";
        return;
    }
    const auto& words = it->second;
    int count = std::count(words.begin(), words.end(), word);
    
    std::cout << "Word: " << word << "\n"
              << "Document: " << doc << "\n"
              << "Count: " << count << "\n"
              << "TF: " << std::fixed << std::setprecision(4) << calc_tf(words, word) << "\n"
              << "TF-IDF: " << std::fixed << std::setprecision(4) << calc_tfidf(corpus, words, word) << "\n";
}

// 3. DOC <document>
void handle_DOC(const Corpus& corpus, const std::string& doc) {
    auto it = corpus.docs.find(doc);
    if (it == corpus.docs.end()) {
        std::cerr << "Document not found: " << doc << "\n";
        return;
    }
    const auto& words = it->second;
    std::set<std::string> unique(words.begin(), words.end());

    std::cout << "Document: " << doc << "\n"
              << "Total words: " << words.size() << "\n"
              << "Unique words: " << unique.size() << "\nTop words:\n";

    // Вычисляем TF-IDF для всех уникальных слов
    std::vector<std::pair<std::string, double>> word_scores;
    std::for_each(unique.begin(), unique.end(), [&](const std::string& w) {
        word_scores.emplace_back(w, calc_tfidf(corpus, words, w));
    });

    // Сортировка по убыванию TF-IDF
    std::sort(word_scores.begin(), word_scores.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Вывод топ-5
    size_t limit = std::min(word_scores.size(), size_t(5));
    size_t idx = 1;
    std::for_each(word_scores.begin(), std::next(word_scores.begin(), limit), [&](const auto& p) {
        std::cout << idx++ << ". " << p.first << "(" 
                  << std::fixed << std::setprecision(4) << p.second << ")\n";
    });
}

// 4. QUERY <w1> <w2> ...
void handle_QUERY(const Corpus& corpus, const std::vector<std::string>& query_words) {
    if (query_words.empty()) return;

    // Вычисляем релевантность каждого документа запросу
    std::vector<std::pair<std::string, double>> results;
    std::for_each(corpus.docs.begin(), corpus.docs.end(), [&](const auto& pair) {
        double score = 0.0;
        // Суммируем TF-IDF по всем словам запроса
        std::for_each(query_words.begin(), query_words.end(), [&](const std::string& w) {
            score += calc_tfidf(corpus, pair.second, w);
        });
        if (score > 0.0) results.emplace_back(pair.first, score);
    });

    // Сортировка по релевантности (убывание)
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Вывод
    std::cout << "Query: ";
    std::for_each(query_words.begin(), query_words.end(), [](const std::string& w) { std::cout << w << " "; });
    std::cout << "\nResults:\n";

    size_t idx = 1;
    std::for_each(results.begin(), results.end(), [&](const auto& p) {
        std::cout << idx++ << ". " << p.first << "(" 
                  << std::fixed << std::setprecision(4) << p.second << ")\n";
    });
}

// Главная точка входа
int main() {
    // 1. Загрузка списка документов
    std::vector<std::string> doc_names = load_document_list("documents.txt");
    if (doc_names.empty()) return 1;

    // 2. Инициализация и загрузка корпуса
    Corpus corpus;
    load_corpus(corpus, doc_names);
    compute_df(corpus);

    // 3. Обработка запросов
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        if (!(iss >> cmd)) continue;

        if (cmd == "WORD") {
            std::string word; iss >> word;
            if (!word.empty()) handle_WORD(corpus, word);
        } 
        else if (cmd == "WORD_IN_DOC") {
            std::string word, doc; iss >> word >> doc;
            if (!word.empty() && !doc.empty()) handle_WORD_IN_DOC(corpus, word, doc);
        } 
        else if (cmd == "DOC") {
            std::string doc; iss >> doc;
            if (!doc.empty()) handle_DOC(corpus, doc);
        } 
        else if (cmd == "QUERY") {
            std::vector<std::string> q_words{
                std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
            handle_QUERY(corpus, q_words);
        } 
        else {
            std::cerr << "Unknown query type: " << cmd << "\n";
        }
        std::cout << "\n"; // Разделитель между выводами запросов
    }

    return 0;
}