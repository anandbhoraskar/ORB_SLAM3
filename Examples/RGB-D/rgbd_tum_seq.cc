/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc < 6)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings (path_to_sequence1 path_to_association1) ... (path_to_sequenceN path_to_associationN) (trajectory_file_name) [prune]" << endl;
        return 1;
    }

    bool bPrunePoints = false;
    if(strcmp("prune", argv[argc-1]) == 0) {
        bPrunePoints = true;
        argc--;
    }

    const int num_seq = (argc-3)/2;
    bool bFileName= (((argc-3) % 2) == 1);
    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]);

    int seq;

    // Retrieve paths to images
    vector <vector<string>> vvstrImageFilenamesRGB;
    vector <vector<string>> vvstrImageFilenamesD;
    vector <vector<double>> vvTimestamps;
    vector <string> vstrAssociationFilename;
    vector<int> nImages;

    //resize the vectors
    vvstrImageFilenamesRGB.resize(num_seq);
    vvstrImageFilenamesD.resize(num_seq);
    vvTimestamps.resize(num_seq);
    vstrAssociationFilename.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;

    for (seq = 0; seq < num_seq; seq++){
        cout << "Loading images for sequence " << seq << "...";
        vstrAssociationFilename[seq] = string(argv[4 + seq*2]);
        LoadImages(vstrAssociationFilename[seq], vvstrImageFilenamesRGB[seq], vvstrImageFilenamesD[seq], vvTimestamps[seq]);

        // Check consistency in the number of images and depthmaps
        nImages[seq] = vvstrImageFilenamesRGB[seq].size();
        if(vvstrImageFilenamesRGB[seq].empty())
        {
            cerr << endl << "No images found in provided path." << endl;
            return 1;
        }
        else if(vvstrImageFilenamesD[seq].size()!=vvstrImageFilenamesRGB[seq].size())
        {
            cerr << endl << "Different number of images for rgb and depth." << endl;
            return 1;
        }

        tot_images += nImages[seq];

        if((nImages[seq]<=0))
        {
            cerr << "ERROR: Failed to load images for sequence" << seq << endl;
            return 1;
        }

    }

    if(bPrunePoints)
        cout << "\n\n\npruning system in use.\n\n\n" << endl;
    else
        cout << "\n\n\npruning system NOT in use\n\n\n" << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    SLAM.setPrunePoints(bPrunePoints);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);


    cv::Mat imRGB, imD;
    int proccIm = 0;

    for (seq = 0; seq<num_seq; seq++)
    {

        cout << endl << "-------" << endl;
        cout << "Start processing sequence " << seq << "..." << endl;
        cout << "Images in the sequence: " << nImages[seq] << endl << endl;

        // Main loop for each sequence
        proccIm = 0;

        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // cout << vvstrImageFilenamesRGB[seq][ni] << endl;
            // Read image and depthmap from file
            imRGB = cv::imread(string(argv[3 + seq*2])+"/"+vvstrImageFilenamesRGB[seq][ni],cv::IMREAD_UNCHANGED);
            imD = cv::imread(string(argv[3 + seq*2])+"/"+vvstrImageFilenamesD[seq][ni],cv::IMREAD_UNCHANGED);
            double tframe = vvTimestamps[seq][ni];

            if(imRGB.empty())
            {
                cerr << endl << "Failed to load image at: "
                    << string(argv[3 + seq*2]) << "/" << vvstrImageFilenamesRGB[seq][ni] << endl;
                return 1;
            }

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Pass the image to the SLAM system
            SLAM.TrackRGBD(imRGB,imD,tframe);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vvTimestamps[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vvTimestamps[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6);
        }

        // // sleep for 10s for BA to complete
        // cout << "Sleeping for 10s" << endl;
        // std::this_thread::sleep_for(std::chrono::microseconds(10000));
        // this doesn't make this thread to sleep...
        
        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }



    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    std::chrono::system_clock::time_point scNow = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(scNow);
    std::stringstream ss;
    ss << now;

    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl;
    cout << "mean tracking time: " << totaltime/proccIm << endl;


    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
