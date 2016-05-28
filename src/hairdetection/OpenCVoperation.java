package hairdetection;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import java.util.List;
import java.util.ArrayList;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
//import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;

public class OpenCVoperation 
{
    //Matrices invovled in algorithm
    private Mat sourceImage;
    private Mat matrix2_grabcut;
    private Mat matrix3_skindetection;
    private Mat orgMask;
    private Mat mask_rgb;
    private Mat newMask;
    private Mat matrix5_grabcut_quantized;
    private Mat matrix6_skin_quantized;
    private Mat matrix7_output;
    private Mat matrix8_max_contour;
    private Mat erosion_dilutionMatrix;
    private Mat matrix9_finalOutput;
    
    //String path names invovled in code
    private final String resultDirectory;
    private final String fileName;
    private final String grabcutOutput;
    private final String skinDetectionOutput;
    private final String grabcut_QuantizedOutput;
    private final String skin_QuantizedOutput;
    private final String finalImage_Output;
    private final String maskOutput;
    private final String maskRgbOutput;
    private final String contourMaskOutput;
    private final String erosionOutput;
    private final String dilationOutput;
    private final String morphingOutput;
    private final String FinalImageWithDilutionOutput;
    public OpenCVoperation(String rootDirectory,String sourceFileName,String[] OutputFileNames)
    {
         resultDirectory = rootDirectory;
         fileName =sourceFileName;
         grabcutOutput = "face2_grabcut.png";
         skinDetectionOutput = "face3_skindetection.png";
         /*--existing file names
         grabcut_QuantizedOutput = "face5_grabcut_quantized.png";
         skin_QuantizedOutput = "face6_skin_quantized.png";
         finalImage_Output = "face7_output.png";
         maskOutput ="skin_mask.png";
         maskRgbOutput = "skin_mask_rgb.png";
         contourMaskOutput="face8_contour_image.png";
         erosionOutput ="erosion_image.png";
         dilationOutput ="dilation_image.png";
         */
         
         dilationOutput ="dilation_image.png";
         maskOutput ="skin_mask.png";
         maskRgbOutput = "skin_mask_rgb.png";
         grabcut_QuantizedOutput = OutputFileNames[2]+".png";
         skin_QuantizedOutput = OutputFileNames[3]+".png";
         morphingOutput = OutputFileNames[4]+".png";
         finalImage_Output = OutputFileNames[5]+".png";
         erosionOutput =OutputFileNames[6]+".png";
         contourMaskOutput=OutputFileNames[7]+".png";
         FinalImageWithDilutionOutput =OutputFileNames[8]+".png";
         
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
    public void testGrabCut()
    {
        sourceImage = Imgcodecs.imread(resultDirectory+fileName,Imgcodecs.CV_LOAD_IMAGE_COLOR);
        System.out.println("result Directory is: "+ resultDirectory+fileName);
        
        Mat result = new Mat(sourceImage.size(),sourceImage.type());
        Mat bgModel = new Mat();    //background model
        Mat fgModel = new Mat();    //foreground model
        
        //draw a rectangle 
        Rect rectangle = new Rect(1,1,sourceImage.cols()-1,sourceImage.rows()-1);
        
        Imgproc.grabCut(sourceImage, result,rectangle, bgModel,fgModel,10,Imgproc.GC_INIT_WITH_RECT);  
        Core.compare(result,new Scalar(3,3,3),result,Core.CMP_EQ);
        matrix2_grabcut = new Mat(sourceImage.size(),CvType.CV_8UC3,new Scalar(255,255,255));
        sourceImage.copyTo(matrix2_grabcut, result);
        
        Imgcodecs.imwrite(resultDirectory+grabcutOutput,matrix2_grabcut);
        
       // displayPixels(matrix2_grabcut);
    }  
    
    //skin detection algorithm 1
    public void skinSegmentation_WithThreshold()
    { 
        //-----code for skin detection--using hsv color method
        
        orgMask= new Mat(matrix2_grabcut.size(),CvType.CV_8UC3);
        orgMask.setTo(new Scalar(0,0,0));
        
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(matrix2_grabcut, hsvImage, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvImage, new Scalar(3,30,50),new Scalar(33,255,255),orgMask);
        
        Imgcodecs.imwrite(resultDirectory + maskOutput, orgMask);
        
        newMask = Imgcodecs.imread(resultDirectory + maskOutput);
        //code for getting rgb skin mask from hsv skin mask
        mask_rgb = new Mat(newMask.size(),CvType.CV_8SC3);
        Imgproc.cvtColor(newMask, mask_rgb, Imgproc.COLOR_HSV2RGB);    
        Imgcodecs.imwrite(resultDirectory+maskRgbOutput, mask_rgb);
        
        //getting only skin image with red background
        matrix3_skindetection= new Mat(sourceImage.size(), sourceImage.type());
        matrix3_skindetection.setTo(new Scalar(0,0,255));
        sourceImage.copyTo(matrix3_skindetection,orgMask);
        
        Imgcodecs.imwrite(resultDirectory+skinDetectionOutput,matrix3_skindetection);
    }
    
    
    // -- skin detection algo 2
    public void skinSegmentation()
    {
        matrix3_skindetection = new Mat(matrix2_grabcut.size(),matrix2_grabcut.type()); 
        matrix3_skindetection.setTo(new Scalar(0,0,255));
        Mat skinMask = new Mat();
        Mat hsvMatrix = new Mat();
        
        Scalar lower = new Scalar(0,48,80);
        Scalar upper = new Scalar(20,255,255);
        
        Imgproc.cvtColor(matrix2_grabcut, hsvMatrix, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvMatrix, lower, upper, skinMask);
        
        Mat kernel =Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(11,11));
        Imgproc.erode(skinMask, skinMask, kernel);
        Imgproc.dilate(skinMask, skinMask, kernel);  
        
        Imgproc.GaussianBlur(skinMask,skinMask, new Size(3,3), 0);
        
        Core.bitwise_and(matrix2_grabcut, matrix2_grabcut, matrix3_skindetection,skinMask);
        Imgcodecs.imwrite(resultDirectory+skinDetectionOutput , matrix3_skindetection);
    }
    
    
      //new skin detection algorithm-- skin detection Algorithm 3
    public void skinDetection2()
    {
        matrix3_skindetection = new Mat(matrix2_grabcut.size(),matrix2_grabcut.type()); 
        matrix3_skindetection.setTo(new Scalar(0,0,255));
        
        Mat  src_YCrCb = new Mat(matrix2_grabcut.size(),CvType.CV_8SC3);
        Mat src_hsv = new Mat(matrix2_grabcut.size(),CvType.CV_8SC3);
        
        Imgproc.cvtColor(matrix2_grabcut, src_YCrCb, Imgproc.COLOR_BGR2YCrCb);  
        matrix2_grabcut.convertTo(src_hsv, CvType.CV_32FC3);
        Imgproc.cvtColor(src_hsv, src_hsv, Imgproc.COLOR_BGR2HSV);
        Core.normalize(src_hsv, src_hsv, 0.00, 255.00, Core.NORM_MINMAX,CvType.CV_32FC3);
        
        for(int r = 0 ; r< matrix2_grabcut.rows(); r++)
        {
            for(int c = 0 ; c< matrix2_grabcut.cols(); c++)
            {
                double[] Pixel_val_rgb = matrix2_grabcut.get(r, c);
                int B= (int)Pixel_val_rgb[0];
                int G= (int)Pixel_val_rgb[1];
                int R= (int)Pixel_val_rgb[2];
                boolean a1= R1(R,G,B);
                
                double[] Pixel_val_YCrCb = src_YCrCb.get(r, c);
                int Y =(int)Pixel_val_YCrCb[0];
                int Cr =(int)Pixel_val_YCrCb[1];
                int Cb =(int)Pixel_val_YCrCb[2];
                boolean a2= R2(Y,Cr,Cb);
                
                double[] Pixel_val_hsv =src_hsv.get(r,c);
                float H = (float)Pixel_val_hsv[0];
                float S = (float)Pixel_val_hsv[1];
                float V = (float)Pixel_val_hsv[2];
                boolean a3= R3(H,S,V);
                
                if(!(a1 && a2 && a3))
                   matrix3_skindetection.put(r, c, new double[]{0,0,255});
                else
                   matrix3_skindetection.put(r,c,sourceImage.get(r, c));
             }
        }
        Imgcodecs.imwrite(resultDirectory+skinDetectionOutput , matrix3_skindetection);
    }
    
    public boolean R1(int R,int G,int B)
    {
       boolean e1 = (R>95) && (G>40) && (B>20) && ((Math.max(R,Math.max(G,B)) - Math.min(R, Math.min(G,B)))>15) && (Math.abs(R-G)>15) && (R>G) && (R>B);
       boolean e2 = (R>220) && (G>210) && (B>170) && (Math.abs(R-G)<=15) && (R>B) && (G>B);
       return (e1||e2);
    }
    
    public boolean R2(float Y, float Cr, float Cb)
    {
        boolean e3 = Cr <= 1.5862*Cb+20;
        boolean e4 = Cr >= 0.3448*Cb+76.2069;
        boolean e5 = Cr >= -4.5652*Cb+234.5652;
        boolean e6 = Cr <= -1.15*Cb+301.75;
        boolean e7 = Cr <= -2.2857*Cb+432.85;
        return e3 && e4 && e5 && e6 && e7;
    }
    public boolean R3(float H, float S, float V) 
    {
        return (H<25) || (H > 230);
    }
    
    
    void setQuantizedImages()
    {
        matrix5_grabcut_quantized = this.quantizeImage(matrix2_grabcut, grabcut_QuantizedOutput);
        matrix6_skin_quantized = this.quantizeImage(matrix3_skindetection,skin_QuantizedOutput);         
    }
      
    public void findContours() 
    {
           
        //Mat orgImage = Imgcodecs.imread(imageFilePath); //load image
        Mat grayImage = new Mat();
        Mat cannyImage= new Mat();
        List<MatOfPoint> contours = new ArrayList<>();  
        Imgproc.cvtColor(this.erosion_dilutionMatrix, grayImage, Imgproc.COLOR_BGR2GRAY);      //bgr to gray scale image conversion
        Imgproc.Canny(grayImage, cannyImage, 100, 200);     //get edges of image
        
        
        //morph edge detected image to improve egde connectivity
        Mat element= Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(5,5));
        Imgproc.morphologyEx(cannyImage, cannyImage, Imgproc.MORPH_CLOSE, element);
        String morphedImagePath = resultDirectory+ morphingOutput;
        Imgcodecs.imwrite(morphedImagePath,cannyImage);
        
        
        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);    //find all contours
        
        matrix8_max_contour = Mat.zeros(cannyImage.size(),CvType.CV_8UC3);
        matrix8_max_contour.setTo(new Scalar(255,255,255));
        double maxArea =Imgproc.contourArea(contours.get(0));
       // double maxArea = 0;
        int maxAreaIndex =0;
        MatOfPoint temp_contour;
        for(int i=1; i<contours.size(); i++)
        {   
            temp_contour = contours.get(i);
            double curr_cont_area = Imgproc.contourArea(temp_contour); 
            if(maxArea < curr_cont_area)
            {
                maxArea=curr_cont_area;
                maxAreaIndex = i;
            }
        }
         //Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),Core.FILLED);
         //   Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),1);
            Imgproc.drawContours(matrix8_max_contour, contours, maxAreaIndex, new Scalar(0,0,0),-1);
            //Imgproc.watershed();
            Imgcodecs.imwrite(resultDirectory+contourMaskOutput,matrix8_max_contour);
         
         
          matrix9_finalOutput = new Mat(sourceImage.size(),sourceImage.type());
         matrix9_finalOutput.setTo(new Scalar(255,255,255));
        
          for( int r =0 ;r < matrix8_max_contour.rows() ; r++)
        {
            for( int c=0; c < matrix8_max_contour.cols() ; c++)
            {
                double[] pixel_val = matrix8_max_contour.get(r, c);
                if(pixel_val[0] == 0 && pixel_val[1] == 0 && pixel_val[2] == 0)
                {
                    double[] orginal_pixel_val = sourceImage.get(r, c);
                    matrix9_finalOutput.put(r,c,orginal_pixel_val);
                }
            }      
        }
          
        //dilution on final image
        int erosion_size=2;
        Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
        Imgproc.dilate(matrix9_finalOutput,matrix9_finalOutput, element2);  
        Imgcodecs.imwrite(resultDirectory+FinalImageWithDilutionOutput, matrix9_finalOutput);
         
        // Imgproc.drawContours(mask, contours, maxAreaIndex, new Scalar(255,255,255));
        
        
        /* 
        //--------------code for copying original image with mask--------------------------
        //Mat croppedImage = new Mat(orgImage.size(),CvType.CV_8UC3);
        Mat croppedImage = new Mat(orgImage.size(),orgImage.type());
        //Mat croppedImage = Mat.ones(orgImage.size(), orgImage.type());
        croppedImage.setTo(new Scalar(0,0,0));      
        String destinationPath = "/home/sujit25/Pictures/croppedImage.png";
        orgImage.copyTo(croppedImage, mask);              // copy original image with mask 
        Imgcodecs.imwrite(destinationPath, croppedImage);
        return destinationPath;
        */
        /* //normalize and save mask
        //Core.normalize(mask, mask,0,255,Core.NORM_MINMAX,CvType.CV_8UC3);
         // write mask 
        String destinationPath2 = "/home/sujit25/Pictures/maskImage.png";
        Imgcodecs.imwrite(destinationPath2,mask);
        return destinationPath2;
        */
        
         /*
        //---------------code for grabcut with final mask------------------------------/
        Mat bgModel = new Mat();        //background model
        Mat fgModel = new Mat();        //foreground model
        Rect rectangle = Imgproc.boundingRect(contours.get(maxAreaIndex));  //draw a rectangle around maximum area contour
        Mat result = new Mat();
        
        Imgproc.grabCut(orgImage, result, rectangle, bgModel, fgModel,1,Imgproc.GC_INIT_WITH_RECT);
        Core.compare(result,new Scalar(3,3,3),result,Core.CMP_EQ);
        Mat foreground = new Mat(orgImage.size(),CvType.CV_8UC3,new Scalar(0,0,255));
        orgImage.copyTo(foreground, result);
        String destination = "/home/sujit25/Pictures/Results/face4_contourImage.png";
        Imgcodecs.imwrite(destination, foreground);
        return destination;
        */
      //  return this.skinSegmentation_WithThreshold(destination);
    }
    
    
    //step5
    public void findImageDifference()
    {
        matrix7_output = new Mat(sourceImage.size(),sourceImage.type());
        matrix7_output.setTo(new Scalar(255,255,255));      //white colored image 
        int rows = sourceImage.rows();
        int cols = sourceImage.cols();
        
        for(int r=0;r <rows ; r++)
        {
            for(int c =0; c < cols; c++)
            {
              //  double grabcut_pixel_val[] =matrix2_grabcut.get(r, c);
              //  double skin_pixel_val[] = newMask.get(r, c);                  
                 double grabcut_pixel_val[] =matrix5_grabcut_quantized.get(r,c);
                 double skin_pixel_val[] =  matrix6_skin_quantized.get(r,c);      
                    //extract those pixels which are non blue in 1st image and red in 2nd image
                  if(  ( (grabcut_pixel_val[0] != 255 ) && (grabcut_pixel_val[1]!=255 ) && (grabcut_pixel_val[2] !=255) )  && ( (skin_pixel_val[0]== 0) && (skin_pixel_val[1]== 0) &&(skin_pixel_val[2]== 255) ) )              
                  {
                       double orgImage_pixel_val[] = sourceImage.get(r, c);
                      //double orgImage_pixel_val[] = new double[]{0,0,0};
                      //double pixel_val[] = new double[3];
                      //pixel_val[0]=pixel_val[1]=pixel_val[2]=0;
                       matrix7_output.put(r, c, orgImage_pixel_val);
                  }
            }
        }       
        Imgcodecs.imwrite(resultDirectory +finalImage_Output, matrix7_output);
        
    }
    public  Mat quantizeImage(Mat image,String destinationFileName)
    {
        //Mat image  = testGrabCut(imageFilePath);
        int rows = image.rows();
        int cols = image.cols();
        Mat newImage = new Mat(image.size(),image.type());
        for(int r = 0 ; r < rows ; r++)
        {
            for(int c =0; c< cols; c++)
            {
                double [] pixel_val = image.get(r, c);
                double [] pixel_data = new double[3];
                pixel_data[0] = reduceVal(pixel_val[0]);
                pixel_data[1] = reduceVal(pixel_val[1]);
                pixel_data[2] = reduceVal(pixel_val[2]);
             //   System.out.print("(" +pixel_data[0]+","+pixel_data[1]+","+pixel_data[2]+") *");
                newImage.put(r, c, pixel_data);
            }
          //  System.out.println();
        }
        /*
        MatOfInt params= new MatOfInt();
        int arr[] = new int[2];
        arr[0]= Imgcodecs.CV_IMWRITE_JPEG_QUALITY;
        arr[1]= 100;
        params.fromArray(arr);
        */
        Imgcodecs.imwrite(resultDirectory+destinationFileName, newImage);
        return newImage;
    }
    
    public double reduceVal(double val)
    {   
        if(val >=0.00 && val <64.00) return 0.00;
        else if(val>=64.00 && val <128.00) return 64.00;
        else if (val>= 128.00 && val < 192.00) return 128.00;
        else return 255.00;
        
    }
    public String getFinalPath()
    {
        return finalImage_Output;
    }
    public  void displayPixels(Mat image)
    {
        for( int r =0 ;r < image.rows() ; r++)
        {
            for( int c=0; c < image.cols() ; c++)
            {
                double[] pixel_val = image.get(r, c);
                System.out.print("(" +pixel_val[0]+","+pixel_val[1]+","+pixel_val[2]+") *");
            }
            System.out.println();
        }
    }
    
    public void performErosion_Dilution()
    {
            erosion_dilutionMatrix = new Mat(this.matrix7_output.size(),this.matrix7_output.type());
            int erosion_size=2;
           
            //erosion
            Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
            Imgproc.erode(matrix7_output, erosion_dilutionMatrix, element1);
            Imgcodecs.imwrite(resultDirectory+erosionOutput,erosion_dilutionMatrix);
            
            /*
            //dilation
            Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,  new Size(2*erosion_size + 1, 2*erosion_size+1));
            Imgproc.dilate(erosion_dilutionMatrix, erosion_dilutionMatrix, element2);
            Imgcodecs.imwrite(resultDirectory+this.dilationOutput,erosion_dilutionMatrix);
            */
    }
    
    public void predict_hair_color()
    {
        Mat hsv_input = matrix9_finalOutput.clone();
        List<Mat> channels= new ArrayList<>();
        Mat hsv_histogram = new Mat();
        MatOfFloat  ranges = new MatOfFloat(0,180);
        MatOfInt histSize = new MatOfInt(255);
        Imgproc.cvtColor(hsv_input, hsv_input, Imgproc.COLOR_BGR2HSV);
        Core.split(hsv_input, channels);
        Imgproc.calcHist(channels.subList(0,1), new MatOfInt(0), new Mat(), hsv_histogram, histSize, ranges);
        int hist_w =256;
        int hist_h = 150;
        
        int bin_w =(int)Math.round(hist_w/histSize.get(0,0)[0]);
        Mat histImage= new Mat(hist_h,hist_w,CvType.CV_8UC3,new Scalar(0,0,0));
        
        for(int i=1;i < histSize.get(0,0)[0];i++)
        {
            Imgproc.line(histImage, new Point(bin_w * (i - 1), hist_h - Math.round(hsv_histogram.get(i - 1, 0)[0])), new Point(bin_w * (i), hist_h - Math.round(hsv_histogram.get(i, 0)[0])), new Scalar(255,0,0),2);			    
        }
        Imgcodecs.imwrite(resultDirectory+"histogram_image.png",histImage);
    }
    
    /*
    public static void main(String[] args)
    {
        OpenCVoperation obj = new OpenCVoperation();
        
        obj.testGrabCut(resultDirectory+fileName);
        obj.skinSegmentation();
        //obj.skinDetection2();
        obj.setQuantizedImages();
        obj.findImageDifference(); 
        obj.performErosion_Dilution();
        obj.findContours();      
    }
    */
}
