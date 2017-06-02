#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>
#include "Ngram.h"

using namespace std;

int main(int argc, char* argv[]){
    // Parse the input argument to the right thing
    string textstr, lmstr, mpstr, parsestr;
    for(int i = 1 ; i < argc ; ++i){
        parsestr = argv[i];
        if(parsestr == "-text")
            textstr = argv[i+1];
        else if(parsestr == "-lm")
            lmstr = argv[i+1];
        else if(parsestr == "-map")
            mpstr = argv[i+1];
    }
    // checkpoint
    //cout << textstr << endl << lmstr << endl << mpstr << endl;
    
    //////////////// get the map /////////////////
    ifstream ifsmap(mpstr.c_str());
    string str;
    map<string, int> wordIndex;
    vector<string> word2vec[13500];
    int wordcount = 0;
    while(getline(ifsmap, str)){
        string Big5 = str.substr(0, 2);
        //if not found in current map, write it into the map and make count ++
        if(wordIndex.count(Big5) == 0)
            wordIndex[Big5] = wordcount++;
        //constructing word2vec map
        int wordIdx = wordIndex[Big5];
        for(int i = 0 ; i < str.size() ; ++i){
            if(str[i] == ' ' || str[i] == '/'){
                string word;
                word = str.substr(i+1, 2);//next Big5 encoded word
                word2vec[wordIdx].push_back(word);//push every word trailing into the wordIdx block of map
            }
        }
    }
    /*checkpoint
    cout << "wordcount: " << wordcount << endl;
    ofstream checkofs("textstr.txt");
    map<string, int>::iterator it;
    for(it = wordIndex.begin(); it != wordIndex.end() ;it++)
        checkofs << it -> first << endl;
    checkofs.close();*/

    /////////////////////////// get the bigram ///////////////////////////
    //parse a bigram module from srilm
    int ngram_order = 2;
    Vocab voc;
    Ngram lm( voc, ngram_order);
    {
        //const char lm_filename[] = "./bigram.lm";
        File lmFile("./bigram.lm", "r");
        lm.read(lmFile);
        lmFile.close();
    }

    /////////////////////////// get the text ////////////////////////////
    //parse the text file
    ifstream ifstext(textstr.c_str());
    string linestr;
    vector<vector<string> > text;
    while(getline(ifstext, linestr)){
        VocabIndex wid;
        vector<string> linetext;
        //look through the line
        for(int i = 0 ; i < linestr.size() ; ++i){
            //find a word(2 conitinued char is not ' ')
            if(linestr[i] != ' ' && linestr[i+1] != ' '){
                string Big5 = linestr.substr(i, 2);
                linetext.push_back(Big5);
                ++i;
            }
        }
        text.push_back(linetext);
    } 
    /*checkpoint
    ofstream checkofs("textstr.txt");
    for(int i = 0 ; i < text.size() ; ++i){
        for(int j = 0 ; j < text[i].size() ; ++j)
            checkofs << text[i][j] << " ";
        checkofs << endl;
    }
    checkofs.close();*/

    ///////////////////// viterbi and output ///////////////////////
    //ofstream checkofs("textstr.txt");
    vector<vector<string> > text_ans;
    for(int i = 0 ; i < text.size() ; ++i){
        //for every line in the text
        vector<vector<string> > candidate;
        vector<vector<double> > delta;
        vector<vector<int> > psi;
        //get all the candidate word into delta to be calculate
        for(int j = 0 ; j < text[i].size() ; ++j){
            string Big5 = text[i][j];
            //find the word in the map(wordIndex)
            int wordIdx = wordIndex.find(Big5)->second;
            vector<string> word_candidate;
            for(int k = 0 ; word2vec[wordIdx][k] != ""; ++k)
                word_candidate.push_back(word2vec[wordIdx][k]);
            candidate.push_back(word_candidate);
        }
        /*
        for(int j = 0 ; j < candidate.size() ; ++j){
            for(int k = 0 ; k < candidate[j].size() ; ++k)
                checkofs << candidate[j][k];
            checkofs << endl;
        }
        checkofs << "\n\n";*/  
        //the first array of delta
        vector<double> delta_axis;
        for(int j = 0 ; j < candidate[0].size() ; ++j){
            double zero = 0;
            delta_axis.push_back(zero);
        }
        delta.push_back(delta_axis);

        //induction of delta and psi
        for(int j = 1 ; j < candidate.size() ; ++j){
            vector<double> delta_axis;
            vector<int> psi_axis;
            //for every candidate word to see the biggest delta bigram 
            for(int k = 0 ; k < candidate[j].size() ; ++k){
                //the testing word is wid
                VocabIndex wid_now= voc.getIndex(candidate[j][k].c_str());
                if(wid_now == Vocab_None)
                        wid_now = voc.getIndex(Vocab_Unknown);
                double biggest = -10000000;
                int arg_biggest = -1;
                for(int l = 0 ; l < candidate[j-1].size() ; ++l){
                    //find the most possible path to go to candidate[j][k](a.k.a. wid)
                    VocabIndex wid_prev = voc.getIndex(candidate[j-1][l].c_str());
                    if(wid_prev == Vocab_None)
                        wid_prev = voc.getIndex(Vocab_Unknown);
                    VocabIndex context[] = {wid_prev, Vocab_None};
                    double prob = lm.wordProb(wid_now, context);
                    if(prob + delta[j-1][l] > biggest){
                        biggest = prob + delta[j-1][l];
                        arg_biggest = l;
                    }
                }
                delta_axis.push_back(biggest);
                psi_axis.push_back(arg_biggest);
            }
            delta.push_back(delta_axis);
            psi.push_back(psi_axis);
        }
        /*checkpoint
        for(int j = 0 ; j < psi.size() ; j ++){
            for(int k = 0 ; k < psi[j].size() ; k++)
                cout << psi[j][k] << " ";
            cout << endl;
        }
        ofstream checkofs("textstr.txt");
        for(int j = 0 ; j < psi.size() ; j++){
            for(int k = 0 ; k < psi[j].size() ; k++)
                checkofs << psi[j][k] << " ";
            checkofs<<endl;
        }
        checkofs.close();
        */
        ///////////////////// backtracking ///////////////////////
        //go to the last axis and find the biggest prob
        double biggest = -1000000;
        int arg_biggest = -1;
        for(int j = 0 ; j < delta[delta.size() - 1].size() ; j++){
            if(delta[delta.size() - 1][j] > biggest){
                biggest = delta[delta.size() - 1][j];
                arg_biggest = j;
            }
        }
        // get all the index of the psi
        vector<int> cand_index;
        cand_index.push_back(arg_biggest);
        /*checkpoint
        cout << "biggest: "<< biggest<<endl;
        cout << "arg_biggest: "<< arg_biggest << endl;*/
        int index = arg_biggest;
        for(int j = psi.size() - 1 ; j >= 0 ; j--){
            int prev_index = psi[j][index];
            cand_index.push_back(prev_index);
            index = prev_index;
        }
        /*checkpoint
        for(int i = cand_index.size() - 1 ; i >= 0 ; i--)
            cout << cand_index[i] << endl;*/
        
        //write back all the word into text_ans
        vector<string> text_ans_line;
        int line_count = 0;
        for(int i = cand_index.size() - 1 ; i >= 0 ; i--){
            string text_line_word = candidate[line_count][cand_index[i]];
            text_ans_line.push_back(text_line_word);
            line_count ++;
        }
        /*checkpoint
        ofstream checkofs("textstr.txt");
        for(int i = 0 ; i < text_ans_line.size() ; i++)
            checkofs << text_ans_line[i] << endl;
        checkofs.close();*/
        text_ans.push_back(text_ans_line);
    }
    /*checkpoint
    ofstream checkofs("textstr.txt");
    for(int i = 0 ; i < text_ans.size();i++){
        for(int j = 0 ; j < text_ans[i].size() ;j++)
            checkofs << text_ans[i][j];
        checkofs << endl;
    }
    checkofs.close();*/
    ///////////////////////// output //////////////////////
    //ofstream ofs("./testdata/1_seg_ans.txt");
	for(int i = 0 ; i < text_ans.size() ; i++){
        cout << "<s>" << " "; 
        for(int j = 0 ; j < text_ans[i].size() ; j++)
            cout << text_ans[i][j] << " ";
        cout << "</s>" << endl;
    }
    //ofs.close();
    ifsmap.close();
    ifstext.close();
    return 0;
}
