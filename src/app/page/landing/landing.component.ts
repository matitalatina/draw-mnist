import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { loadLayersModel, LayersModel, tensor, Tensor, Rank } from '@tensorflow/tfjs';
import { MatSliderChange } from '@angular/material/slider';
import * as handTrack from 'handtrackjs';
import { zoomInOnEnterAnimation, fadeOutOnLeaveAnimation, tadaOnEnterAnimation } from 'angular-animations';

interface Slider {
  min: number;
  max: number;
  value: number;
}

interface HandPrediction {
  bbox: [number, number, number, number]; // x, y, width, height
  class: 'hand';
  score: number;
}

const handModelParams = {
  flipHorizontal: true,   // flip e.g for video
  imageScaleFactor: 0.5,  // reduce input image size for gains in speed.
  maxNumBoxes: 2,        // maximum number of boxes to detect
  iouThreshold: 0.5,      // ioU threshold for non-max suppression
  scoreThreshold: 0.7,    // confidence threshold for predictions.
};

const SLIDER_THRESHOLD = 3.5;
const HAND_SCORE_THRESHOLD = 0.6;

@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.scss'],
  animations: [
    zoomInOnEnterAnimation(),
    fadeOutOnLeaveAnimation(),
    tadaOnEnterAnimation({ delay: 1000 }),
  ],
})
export class LandingComponent implements OnInit {
  decoder: LayersModel;
  handModel;
  isVideo = false;
  sliders: Slider[] = [...Array(4)].map(() => ({ min: -SLIDER_THRESHOLD, max: SLIDER_THRESHOLD, value: 0 }));
  isLoading = false;
  @ViewChild('canvasDigit', { static: false }) canvas: ElementRef<HTMLCanvasElement>;
  @ViewChild('canvasHand', { static: false }) canvasHand: ElementRef<HTMLCanvasElement>;
  @ViewChild('webcam', { static: false }) video: ElementRef<HTMLVideoElement>;

  constructor() {

  }

  ngOnInit() {
    this.isLoading = true;
    Promise.all([
      this.loadModel(),
      handTrack.load(handModelParams).then(m => { console.log('Model loaded'); this.handModel = m; }),
    ]).then(() => {
      this.isLoading = false;
      setTimeout(() => this.predict(), 0);
    });
  }

  async loadModel() {
    this.decoder = await loadLayersModel('assets/models/mnist_decoder_z4.json');
  }

  slideSlider(slider: Slider, event: MatSliderChange) {
    slider.value = event.value;
    this.predict();
  }

  predict() {
    const pixelSize = 5;
    const z = this.sliders.map(s => s.value);
    const prediction = (this.decoder.predict(tensor([z])) as Tensor<Rank>).arraySync()[0];

    const canvasWidth = 28;
    const canvasHeight = 28;
    const ctx: CanvasRenderingContext2D = this.canvas.nativeElement.getContext('2d');

    for (let y = 0; y < canvasHeight; ++y) {
      for (let x = 0; x < canvasWidth; ++x) {
        const value = prediction[y * canvasWidth + x] * 255;
        ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
        ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
      }
    }
  }

  startVideo() {
    handTrack.startVideo(this.video.nativeElement).then((status) => {
      console.log('video started', status);
      if (status) {
        console.log('Video started. Now tracking');
        this.isVideo = true;
        this.runDetection();
      } else {
        console.log('Please enable video');
      }
    });
  }

  toggleVideo() {
    if (!this.isVideo) {
      console.log('Starting video');
      this.startVideo();
    } else {
      console.log('Stopping video');
      handTrack.stopVideo(this.video.nativeElement);
      this.isVideo = false;
      console.log('Video stopped');
    }
  }

  runDetection() {
    this.handModel.detect(this.video.nativeElement).then((predictions: HandPrediction[]) => {
      const canvasWidth = this.canvasHand.nativeElement.width;
      const canvasHeight = this.canvasHand.nativeElement.height;
      predictions
        .filter(p => p.score >= HAND_SCORE_THRESHOLD)
        .slice(0, 2).forEach((p, i) => {
          const [x, y, width, height] = p.bbox;
          const newLatentX = (x + width / 2 - canvasWidth / 2) / canvasWidth * SLIDER_THRESHOLD * 2.3;
          const newLatentY = (y + height / 2 - canvasHeight / 2) / canvasHeight * SLIDER_THRESHOLD * 2.3;
          const [x1, y1, x2, y2] = this.sliders.map(s => s.value);
          const sliderDistanceFromHand = [[x1, y1], [x2, y2]]
            .map(([sliderX, sliderY]) => Math.hypot(sliderX - newLatentX, sliderY - newLatentY));
          const startSliderIndex = sliderDistanceFromHand[0] < sliderDistanceFromHand[1] ? 0 : 2;
          this.sliders[startSliderIndex] = {
            ...this.sliders[startSliderIndex],
            value: newLatentX
          };
          this.sliders[startSliderIndex + 1] = {
            ...this.sliders[startSliderIndex + 1],
            value: newLatentY
          };
        });
      const canvas = this.canvasHand.nativeElement;
      this.handModel.renderPredictions(predictions, canvas, canvas.getContext('2d'), this.video.nativeElement);
      this.predict();
      if (this.isVideo) {
        requestAnimationFrame(() => this.runDetection());
      }
    });
  }

}

