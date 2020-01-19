import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { loadLayersModel, LayersModel, tensor, Tensor, Rank } from '@tensorflow/tfjs';
import { select } from 'd3';
import { MatSliderChange } from '@angular/material/slider';

interface Slider {
  min: number;
  max: number;
  value: number;
}

@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.scss']
})
export class LandingComponent implements OnInit {
  decoder: LayersModel;
  sliders: Slider[] = [...Array(4)].map(() => ({ min: -3, max: 3, value: 0 }));
  @ViewChild('canvasDigit', { static: false }) canvas: ElementRef;
  constructor() {

  }

  ngOnInit() {
    this.loadModel().then(() => this.predict());
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
}
