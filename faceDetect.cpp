/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Functions for finding faces and drawing boxes around them

  The path to the Haar cascade file is define in faceDetect.h
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <opencv2/opencv.hpp>
#include "faceDetect.h"

namespace {

std::string faceCascadePath() {
    namespace fs = std::filesystem;
    const char* e = std::getenv("RISO_FACE_CASCADE");
    if (e != nullptr && e[0] != '\0' && fs::exists(e)) {
        return std::string(e);
    }
    static constexpr const char* kCandidates[] = {
        "./haarcascade_frontalface_alt2.xml",
        "../haarcascade_frontalface_alt2.xml",
    };
    for (const char* p : kCandidates) {
        if (fs::exists(p)) {
            return std::string(p);
        }
    }
    return "./haarcascade_frontalface_alt2.xml";
}

}  // namespace

/*
  Arguments:
  cv::Mat grey  - a greyscale source image in which to detect faces
  std::vector<cv::Rect> &faces - a standard vector of cv::Rect rectangles indicating where faces were found
     if the length of the vector is zero, no faces were found
 */
int detectFaces( cv::Mat &grey, std::vector<cv::Rect> &faces ) {
  // a static variable to hold a half-size image
  static cv::Mat half;
  
  // a static variable to hold the classifier
  static cv::CascadeClassifier face_cascade;
  static bool loadAttempted = false;
  static bool loadOk = false;

  if (!loadAttempted) {
    loadAttempted = true;
    loadOk = face_cascade.load(faceCascadePath());
    if (!loadOk) {
      printf("Unable to load face cascade (set RISO_FACE_CASCADE). Colorful-faces filter will skip detection.\n");
    }
  }
  if (!loadOk) {
    faces.clear();
    return 0;
  }

  // clear the vector of faces
  faces.clear();
  
  // cut the image size in half to reduce processing time
  cv::resize( grey, half, cv::Size(grey.cols/2, grey.rows/2) );

  // equalize the image
  cv::equalizeHist( half, half );

  // apply the Haar cascade detector
  face_cascade.detectMultiScale( half, faces );

  // adjust the rectangle sizes back to the full size image
  for(int i=0;i<faces.size();i++) {
    faces[i].x *= 2;
    faces[i].y *= 2;
    faces[i].width *= 2;
    faces[i].height *= 2;
  }

  return(0);
}

/* Draws rectangles into frame given a vector of rectangles
   
   Arguments:
   cv::Mat &frame - image in which to draw the rectangles
   std::vector<cv::Rect> &faces - standard vector of cv::Rect rectangles
   int minSize - ignore rectangles with a width small than this argument
   float scale - scale the rectangle values by this factor (in case frame is different than the source image)
 */
int drawBoxes( cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth, float scale  ) {
  // The color to draw, you can change it here (B, G, R)
  cv::Scalar wcolor(170, 120, 110);

  for(int i=0;i<faces.size();i++) {
    if( faces[i].width > minWidth ) {
      cv::Rect face( faces[i] );
      face.x *= scale;
      face.y *= scale;
      face.width *= scale;
      face.height *= scale;
      cv::rectangle( frame, face, wcolor, 3 );
    }
  }

  return(0);
}
