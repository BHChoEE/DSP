#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include "hmm.h"

using namespace std;

int main(int argc, char** argv){
    //================= parsing =================
    //getting test data
    string line;
    ifstream test_file;   
    test_file.open(argv[2]);
    vector<int*> testData;
    while(test_file >> line){
         int* tmp = new int[line.length()];
         for(int i = 0; i < line.length() ; ++i)
            tmp[i] = line.c_str()[i] - 'A';
         testData.push_back(tmp);
    }
    test_file.close();
    /*test for parsing
    for(int i = 0 ; i < 50 ; i++)
        cout << testData[2499][i] << " " ;
    cout << endl;*/

    //================== testing =================

    ifstream modellist_file;
    modellist_file.open(argv[1]);
    vector<string> modellist;
    while(modellist_file>> line)
        modellist.push_back(line);

    int model_ans[2500] = {0};
    double model_prob_ans[2500] = {0};

    modellist_file.close();
    vector<HMM*> hmm;
    for(int i = 0 ; i < modellist.size() ; ++i){
        HMM* hmm_tmp = new HMM;
        loadHMM(hmm_tmp, modellist[i].c_str());
        hmm.push_back(hmm_tmp);
    }
    /* check if hmm is correct
    for(int m = 0 ; m < hmm.size() ; ++m){
        cout << "model_0" << m << endl;
        for(int i = 0 ; i < 6 ; i++)
            cout << hmm[m] -> initial[i] << " ";
        cout << endl;
        for(int i = 0 ; i < 6 ; i++){
            for(int j = 0 ; j < 6 ; ++j)
                cout << hmm[m] -> transition[i][j] << " ";
            cout << endl;
        }
        cout << endl;
        for(int i = 0 ; i < 6 ; i++){
            for(int j = 0 ; j < 6 ; ++j)
                cout << hmm[m] -> observation[i][j] << " ";
            cout << endl;
        }
        cout << endl;
    }*/



for(int line = 0 ; line < 2500 ; line++){
    //every sample
    double maxprob = 0;
    int argmax_model = 0; 
    for(int model = 0 ; model < modellist.size() ; model++){
        //load HMM
        HMM* hmm1 = hmm[model];

        double delta[50][6] = {0};
        int psi[50][6] = {0};
        //construct delta
        //alpha initialization
        for(int i = 0 ; i < 6 ; ++i){
            delta[0][i] = hmm1 -> initial[i] * hmm1 -> observation[testData[line][0]][i];
            //cout << delta[0][i] << " ";
        }
        //cout << endl;
        //alpha recursion    
        for(int t = 0 ; t < 49 ; ++t)
            for(int j = 0 ; j < 6 ; ++j){
                double max = 0;
                for(int i = 0 ; i < 6 ; ++i){
                    if(delta[t][i]* hmm1 -> transition[i][j] > max)
                        max = delta[t][i]* hmm1 -> transition[i][j];    
                }
                delta[t+1][j] = max * hmm1 -> observation[testData[line][t]][j];
            }
        /*
        for(int i = 0 ; i < 50 ; i++){
            for(int j = 0 ; j < 6 ; j++)
                cout << delta[i][j] << " ";
            cout << endl;
        }
        cout << endl;
        for(int i = 0 ; i < 50 ; i++){
            for(int j = 0 ; j < 6 ; j++)
                cout << psi[i][j] << " ";
            cout << endl;
        }*/
        //alpha termination
        double max =  0 ;
        for(int i = 0 ; i < 6 ; ++i){ 
            if(delta[49][i] > max)
                max = delta[49][i];
        //cout << delta[49][i];
        }
        //cout << max << endl;
        if(max > maxprob){
            maxprob = max;
            argmax_model = model;
        }
    }
    
    model_ans[line] = argmax_model;
    model_prob_ans[line] = maxprob;
    cout << line << endl;
    cout << maxprob << " " << argmax_model << endl;
}

    // output file to reslut.txt
    ofstream result_file;
    result_file.open(argv[3]);
    for(int i = 0 ; i < 2500 ; ++i){
        result_file << "model_0" << model_ans[i] + 1 <<  ".txt ";
        result_file <<  model_prob_ans[i] << endl;
    }
    result_file.close();

    // get accuracy
    ifstream ans_file;
    vector<int> ans;
    ans_file.open("../testing_answer.txt");
    while(ans_file >> line){
        int tmp;
        if(line.c_str()[7] == '1'){
            tmp = 0;
            ans.push_back(tmp);
        }
        else if(line.c_str()[7] == '2'){
            tmp = 1;
            ans.push_back(tmp);
        }
        else if(line.c_str()[7] == '3'){
            tmp = 2;
            ans.push_back(tmp);
        }
        else if(line.c_str()[7] == '4'){
            tmp = 3;
            ans.push_back(tmp);
        }
        else if(line.c_str()[7] == '5'){
            tmp = 4;
            ans.push_back(tmp);
        }
    }
    ans_file.close();
    double correct = 0;
    for(int i = 0 ; i < 2500 ; ++i){
        if(model_ans[i] == ans[i])
            correct += 1;
    }
    cout << correct / 25 << '%' << endl;
    //output accuracy
    ofstream acc_file;
    acc_file.open("acc.txt");
    acc_file << correct / 2500 << endl;
    acc_file.close();
    return 0;
}