/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hairdetection;

import java.io.File;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class FXMLDocumentController implements Initializable {
    
    @FXML private ImageView imgStep1;
    @FXML private ImageView imgStep2;
    @FXML private ImageView imgStep3;
    @FXML private ImageView imgStep4;
    @FXML private ImageView imgStep5;
    @FXML private ImageView imgStep6;
    @FXML private ImageView imgStep7;
    @FXML private ImageView imgStep8;
    
    Image img1;
    Image img2;
    Image img3;
    Image img4;
    Image img5;
    Image img6;
    Image img7;
    Image img8; 
    
    File imgFile1;
    File imgFile2;
    File imgFile3;
    File imgFile4;
    File imgFile5;
    File imgFile6;
    File imgFile7;
    File imgFile8;
    
    OpenCVoperation obj;
    FileChooser fileChooser = new FileChooser();
    String path;
    
    String[] results = {"filler0", "filler1", "face2_grabcut_quantized", "face3_skindetection_quantized", "face4_morphedImage", "face5_difference", "face6_erosion", "face7_contour", "face8_dilution" };
      
    FXMLLoader fxmlloader = new FXMLLoader();
    
    @FXML
    private void handleLoadAction(ActionEvent event) {
        fileChooser.setTitle("Open an Image");
        FileChooser.ExtensionFilter extFilter = new FileChooser.ExtensionFilter("Image files (*.jpeg,*.jpg, *.png) ","*.jpeg","*.jpg", "*.png");
        fileChooser.getExtensionFilters().add(extFilter);
        imgFile1 = fileChooser.showOpenDialog(new Stage());;
        img1 = new Image(imgFile1.toURI().toString());
        imgStep1.setImage(img1);
        //get parent directory
        path = imgFile1.getParent();
        
        System.out.println(imgFile1.toURI().toString());
    }
    
    @FXML
    private void handleGenerateAction(ActionEvent event)
    {
       // System.out.println("File name is:" +imgFile1.getName());
        
        obj = new OpenCVoperation(path+"/",imgFile1.getName(),results);
        obj.testGrabCut();
        //obj.skinSegmentation_WithThreshold();
        obj.skinSegmentation();
        //obj.skinDetection2();
        obj.setQuantizedImages();
        obj.findImageDifference();
        obj.performErosion_Dilution();
        obj.findContours();
        obj.predict_hair_color();
        
        //Generate file to load images
        imgFile2 = new File(path,results[2]+".png");        //grabcut quantized
        imgFile3 = new File(path,results[3]+".png");        //skin detection quantized
        imgFile4 = new File(path,results[4]+".png");        //morphed Image
        imgFile5 = new File(path,results[5]+".png");        //face difference o/p
        imgFile6 = new File(path,results[6]+".png");        //face erosion
        imgFile7 = new File(path,results[7]+".png");        //face contour
        imgFile8 = new File(path,results[8]+".png");        //face dilution
        
        
        //Generate images using files
        img2 = new Image(imgFile2.toURI().toString());
        img3 = new Image(imgFile3.toURI().toString());
        img4 = new Image(imgFile4.toURI().toString());
        img5 = new Image(imgFile5.toURI().toString());
        img6 = new Image(imgFile6.toURI().toString());
        img7 = new Image(imgFile7.toURI().toString());
        img8 = new Image(imgFile8.toURI().toString());
        
        //Load images in Output Window
        imgStep2.setImage(img2); 
        imgStep3.setImage(img3); 
        imgStep4.setImage(img4); 
        imgStep5.setImage(img5); 
        imgStep6.setImage(img6); 
        imgStep7.setImage(img7); 
        imgStep8.setImage(img8); 
        
    }
    
    @Override
    public void initialize(URL url, ResourceBundle rb) 
    {
        // TODO
    }    
    
}
