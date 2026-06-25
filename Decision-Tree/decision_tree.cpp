#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// Using enums instead of strings for blazing fast comparisons
enum class Weather { Sunny, Rainy };
enum class Temp    { Hot, Mild, Cool };

// Structure representing a single row in your dataset
struct BeachSample {
    Weather weather;
    Temp temperature;
    bool isWeekend;
    bool goToBeach; // Target Class
};

// Node structure for our Decision Tree
struct Node {
    bool isLeaf = false;
    bool prediction = false; // Only matters if isLeaf is true
    
    std::string splitFeature = "";
    Weather weatherThreshold;
    Temp tempThreshold;
    bool boolThreshold;
    
    Node* left = nullptr;  // True branch
    Node* right = nullptr; // False branch
};

// Helper to calculate Gini Impurity
double calculateGini(int yesCount, int noCount) {
    int total = yesCount + noCount;
    if (total == 0) return 0.0;
    
    double pYes = static_cast<double>(yesCount) / total;
    double pNo = static_cast<double>(noCount) / total;
    
    return 1.0 - (pYes * pYes) - (pNo * pNo);
}

// Helper to count target distributions
void getTargetCounts(const std::vector<BeachSample>& dataset, int& yesCount, int& noCount) {
    yesCount = 0; noCount = 0;
    for (const auto& sample : dataset) {
        if (sample.goToBeach) yesCount++;
        else noCount++;
    }
}

// Evaluates every feature to find the split that results in the lowest Gini Impurity
void findBestSplit(const std::vector<BeachSample>& dataset, std::string& bestFeature, 
                   Weather& bestWeather, Temp& bestTemp, bool& bestBool, double& minGini) {
    
    minGini = 999.0;
    int totalSamples = dataset.size();

    // 1. Evaluate "Weather" Split (Sunny vs Rainy)
    {
        int leftYes = 0, leftNo = 0, rightYes = 0, rightNo = 0;
        for (const auto& s : dataset) {
            if (s.weather == Weather::Sunny) { s.goToBeach ? leftYes++ : leftNo++; }
            else { s.goToBeach ? rightYes++ : rightNo++; }
        }
        double totalGini = ((double)(leftYes + leftNo) / totalSamples) * calculateGini(leftYes, leftNo) +
                           ((double)(rightYes + rightNo) / totalSamples) * calculateGini(rightYes, rightNo);
        if (totalGini < minGini) {
            minGini = totalGini; bestFeature = "Weather"; bestWeather = Weather::Sunny;
        }
    }

    // 2. Evaluate "Temperature" Splits
    // Since Temp has 3 categories (Hot, Mild, Cool), we test splits separating one group from the rest
    Temp tempOptions[] = {Temp::Hot, Temp::Mild, Temp::Cool};
    for (Temp t : tempOptions) {
        int leftYes = 0, leftNo = 0, rightYes = 0, rightNo = 0;
        for (const auto& s : dataset) {
            if (s.temperature == t) { s.goToBeach ? leftYes++ : leftNo++; }
            else { s.goToBeach ? rightYes++ : rightNo++; }
        }
        double totalGini = ((double)(leftYes + leftNo) / totalSamples) * calculateGini(leftYes, leftNo) +
                           ((double)(rightYes + rightNo) / totalSamples) * calculateGini(rightYes, rightNo);
        if (totalGini < minGini) {
            minGini = totalGini; bestFeature = "Temperature"; bestTemp = t;
        }
    }

    // 3. Evaluate "Weekend" Split (Yes vs No)
    {
        int leftYes = 0, leftNo = 0, rightYes = 0, rightNo = 0;
        for (const auto& s : dataset) {
            if (s.isWeekend) { s.goToBeach ? leftYes++ : leftNo++; }
            else { s.goToBeach ? rightYes++ : rightNo++; }
        }
        double totalGini = ((double)(leftYes + leftNo) / totalSamples) * calculateGini(leftYes, leftNo) +
                           ((double)(rightYes + rightNo) / totalSamples) * calculateGini(rightYes, rightNo);
        if (totalGini < minGini) {
            minGini = totalGini; bestFeature = "Weekend"; bestBool = true;
        }
    }
}

// Recursive function to build the tree architecture
Node* buildTree(const std::vector<BeachSample>& dataset, int depth, int maxDepth) {
    Node* node = new Node();
    
    int yesCount = 0, noCount = 0;
    getTargetCounts(dataset, yesCount, noCount);
    
    if (yesCount == 0 || noCount == 0 || depth >= maxDepth || dataset.empty()) {
        node->isLeaf = true;
        node->prediction = (yesCount >= noCount);
        return node;
    }

    std::string bestFeature = "";
    Weather bestWeather = Weather::Sunny;
    Temp bestTemp = Temp::Hot;
    bool bestBool = true;
    double minGini = 999.0;
    
    findBestSplit(dataset, bestFeature, bestWeather, bestTemp, bestBool, minGini);

    if (minGini >= calculateGini(yesCount, noCount)) {
        node->isLeaf = true;
        node->prediction = (yesCount >= noCount);
        return node;
    }

    node->splitFeature = bestFeature;
    std::vector<BeachSample> leftSubset, rightSubset;

    if (bestFeature == "Weather") {
        node->weatherThreshold = bestWeather;
        for (const auto& s : dataset) {
            if (s.weather == bestWeather) leftSubset.push_back(s);
            else rightSubset.push_back(s);
        }
    } else if (bestFeature == "Temperature") {
        node->tempThreshold = bestTemp;
        for (const auto& s : dataset) {
            if (s.temperature == bestTemp) leftSubset.push_back(s);
            else rightSubset.push_back(s);
        }
    } else if (bestFeature == "Weekend") {
        node->boolThreshold = bestBool;
        for (const auto& s : dataset) {
            if (s.isWeekend == bestBool) leftSubset.push_back(s);
            else rightSubset.push_back(s);
        }
    }

    node->left = buildTree(leftSubset, depth + 1, maxDepth);
    node->right = buildTree(rightSubset, depth + 1, maxDepth);
    return node;
}

// Helper string mappers for printing out the console tree cleanly
std::string weatherToString(Weather w) { return (w == Weather::Sunny) ? "Sunny" : "Rainy"; }
std::string tempToString(Temp t) {
    if (t == Temp::Hot) return "Hot";
    if (t == Temp::Mild) return "Mild";
    return "Cool";
}

void printTree(Node* root, int indent = 0) {
    if (root == nullptr) return;
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    
    if (root->isLeaf) {
        std::cout << "-> [LEAF PREDICTION: GO TO BEACH = " << (root->prediction ? "YES" : "NO") << "]\n";
    } else {
        std::cout << "|-- Condition: [" << root->splitFeature;
        if (root->splitFeature == "Weather") std::cout << " == " << weatherToString(root->weatherThreshold);
        else if (root->splitFeature == "Temperature") std::cout << " == " << tempToString(root->tempThreshold);
        else if (root->splitFeature == "Weekend") std::cout << " == Yes";
        std::cout << "]\n";
        
        printTree(root->left, indent + 1);
        printTree(root->right, indent + 1);
    }
}

int main() {
    // Exact Dataset from your prompt loaded systematically
    std::vector<BeachSample> dataset = {
        {Weather::Sunny, Temp::Hot,  true,  true},  // Sunny, Hot,  Yes -> Yes
        {Weather::Sunny, Temp::Hot,  false, true},  // Sunny, Hot,  No  -> Yes
        {Weather::Rainy, Temp::Mild, true,  false}, // Rainy, Mild, Yes -> No
        {Weather::Sunny, Temp::Cool, true,  false}, // Sunny, Cool, Yes -> No
        {Weather::Rainy, Temp::Hot,  false, false}, // Rainy, Hot,  No  -> No
        {Weather::Sunny, Temp::Hot,  true,  true},  // Sunny, Hot,  Yes -> Yes
        {Weather::Sunny, Temp::Mild, false, true}   // Sunny, Mild, No  -> Yes
    };

    std::cout << "--- Training Decision Tree On Beach Dataset ---\n";
    Node* root = buildTree(dataset, 0, 4);
    printTree(root);
    std::cout << "-----------------------------------------------\n";

    return 0;
}