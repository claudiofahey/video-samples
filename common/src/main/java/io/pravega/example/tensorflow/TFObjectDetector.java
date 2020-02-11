/*
 * Copyright (c) 2019 Dell Inc., or its subsidiaries. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 */
package io.pravega.example.tensorflow;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.*;

import java.io.InputStream;
import java.io.Serializable;
import java.nio.FloatBuffer;
import java.util.List;

/**
 * ObjectDetector class to detect objects using pre-trained models with TensorFlow Java API.
 */
public class TFObjectDetector implements Serializable {
    public static final String JPEG_BYTES_PLACEHOLDER_NAME = "image";
    private final Logger log = LoggerFactory.getLogger(TFObjectDetector.class);

//    public List<Recognition> recognitions = null;
//    Graph graph;
    Session session;
    // Params used for image processing
    int IMAGE_DIMENSION = 416;
    float SCALE = 255f;
    Output<Float> imagePreprocessingOutput;
    private List<String> LABEL_DEF;
    private static TFObjectDetector single_instance = null;


    public static TFObjectDetector getInstance() {
        if(single_instance == null) {
            single_instance = new TFObjectDetector();
        }

        return single_instance;
    }

    public TFObjectDetector() {
        log.info("@@@@@@@@@@@  new TF @@@@@@@@@@@  " );

        InputStream graphFile = getClass().getResourceAsStream("/tiny-yolo-voc.pb");       // The model
        InputStream labelFile = getClass().getResourceAsStream("/yolo-voc-labels.txt");    // labels for classes used to train model

        byte[] GRAPH_DEF = IOUtil.readAllBytesOrExit(graphFile);
        LABEL_DEF = IOUtil.readAllLinesOrExit(labelFile);
        Graph graph = new Graph();
        graph.importGraphDef(GRAPH_DEF);
        session = new Session(graph);
        GraphBuilder graphBuilder = new GraphBuilder(graph);

        imagePreprocessingOutput =
                graphBuilder.div( // Divide each pixels with the MEAN
                        graphBuilder.resizeBilinear( // Resize using bilinear interpolation
                                graphBuilder.expandDims( // Increase the output tensors dimension
                                        graphBuilder.decodeJpeg(
                                                graph.opBuilder("Placeholder", JPEG_BYTES_PLACEHOLDER_NAME)
                                                        .setAttr("dtype", DataType.STRING)
                                                        .build().output(0), 3),
                                        graphBuilder.constant("make_batch", 0)),
                                graphBuilder.constant("size", new int[]{IMAGE_DIMENSION, IMAGE_DIMENSION})),
                        graphBuilder.constant("scale", SCALE));
    }

    /**
     * Detect objects on the given image
     *
     * @param image the location of the image
     * @return output image with objects detected
     */
    public byte[] detect(byte[] image) {
        long start = System.currentTimeMillis();
        byte[] finalData = null;

        // leak BEGIN
        final float[] tensorFlowOutput = executeYOLOGraph(image);
        // leak END
//        List<Recognition> recognitions = YOLOClassifier.getInstance().classifyImage(tensorFlowOutput, LABEL_DEF);
//        List<Recognition> recognitions = new ArrayList();

//        recognitions.add(
//            new Recognition(99, "none", 1.0f,
//                new BoxPosition(50f, 50f, 100f, 100f)));
//
//        log.info("recognitions={}", recognitions);
//
//        finalData = ImageUtil.getInstance().labelImage(image, recognitions);
        finalData = image;
        long end = System.currentTimeMillis();
        log.info("@@@@@@@@@@@  TENSORFLOW  TIME TAKEN FOR DETECTION @@@@@@@@@@@  " + (end - start));

        return finalData;
    }

    /**
     * Executes graph on the given preprocessed image
     *
     * @param jpegBytes JPEG image
     * @return output tensor returned by tensorFlow
     */
    private float[] executeYOLOGraph(byte[] jpegBytes) {
        // Preprocess image (decode JPEG and resize)
        try (final Tensor<?> jpegTensor = Tensor.create(jpegBytes)) {
            final List<Tensor<?>> imagePreprocessorOutputs = session
                    .runner()
                    .feed(JPEG_BYTES_PLACEHOLDER_NAME, jpegTensor)
                    .fetch(imagePreprocessingOutput.op().name())
                    .run();
            assert imagePreprocessorOutputs.size() == 1;
            try (final Tensor<Float> preprocessedImageTensor = imagePreprocessorOutputs.get(0).expect(Float.class)) {
                // YOLO object detection
                final List<Tensor<?>> detectorOutputs = session
                        .runner()
                        .feed("input", preprocessedImageTensor)
                        .fetch("output")
                        .run();
                assert detectorOutputs.size() == 1;
                try (final Tensor<Float> resultTensor = detectorOutputs.get(0).expect(Float.class)) {
                    final float[] outputTensor = new float[YOLOClassifier.getInstance().getOutputSizeByShape(resultTensor)];
                    final FloatBuffer floatBuffer = FloatBuffer.wrap(outputTensor);
                    resultTensor.writeTo(floatBuffer);
                    return outputTensor;
                }
            }
        }
    }

//    public List<Recognition> getRecognitions() {
//        return recognitions;
//    }

//    private static void printToConsole() {
//        for (Recognition recognition : getRecognitions()) {
//            System.out.println("TITLE :  " + recognition.getTitle());
//            System.out.println("CONFIDENCE :  " + recognition.getConfidence());
//        }
//    }

//    private void setRecognitions(List<Recognition> recs) {
//        this.recognitions = recs;
//    }

}