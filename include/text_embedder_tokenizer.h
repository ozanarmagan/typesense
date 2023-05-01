#pragma

#include <vector>
#include <unordered_map>
#include <sentencepiece_processor.h>
#include <tokenizer/bert_tokenizer.hpp>


enum class TokenizerType {
    bert,
    distilBert,
    xlm_roberta
};

struct encoded_input_t {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> token_type_ids;
    std::vector<int64_t> attention_mask;
};



// Create a base class for all tokenizers to inherit from
class TextEmbeddingTokenizer {
    private:
    public:
        virtual encoded_input_t Encode(const std::string& text) = 0;
};

class BertTokenizerWrapper : public TextEmbeddingTokenizer {
    protected:
        std::unique_ptr<BertTokenizer> bert_tokenizer_;
    public:
        BertTokenizerWrapper(const std::string& vocab_path);
        encoded_input_t Encode(const std::string& text) override;
};

class DistilBertTokenizer : public BertTokenizerWrapper {
    public:
        DistilBertTokenizer(const std::string& vocab_path);
        encoded_input_t Encode(const std::string& text) override;
};

class XLMRobertaTokenizer : public TextEmbeddingTokenizer {
    private:
        inline static constexpr uint8_t fairseq_offset = 1;
        std::unordered_map<std::string, int64_t> fairseq_tokens_to_ids_ = {
            {"<s>", 0},
            {"<pad>", 1},
            {"</s>", 2},
            {"<unk>", 3},
        };
        std::unique_ptr<sentencepiece::SentencePieceProcessor> sentencepiece_tokenizer_;
        const int token_to_id(const std::string& token);
        const std::vector<std::string> tokenize(const std::string& text);
    public:
        XLMRobertaTokenizer(const std::string& model_path);
        encoded_input_t Encode(const std::string& text) override;
};