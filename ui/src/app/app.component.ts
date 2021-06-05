import { Component } from '@angular/core';
import * as $ from "jquery";

declare const CanvasJS: any;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ui';
  
  ngOnInit() {
    let dataPoints: { x:any; y: any; }[] = [];
    let dpsLength = 0;
    let last_n_prices = 100;
    let scale_factor = 10000;
    let scaleBreak = 142;

    let chart = new CanvasJS.Chart("chartContainer", {
      zoomEnabled: true,
		  animationEnabled: true,
      exportEnabled: true,
      title: {
        text: "Test Chart"
      },
      data: [{
        type: "line",
        dataPoints: dataPoints,
      }],
      axisY: {
        prefix: (1/scale_factor) + "x",
        scaleBreaks: {
          // auoCalculate: true
          customBreaks: [{
            startValue: 0,
            endValue: scaleBreak,
            color:"green"
          }]
        }
      }
    });

    $.getJSON("http://localhost:1234/services/last/" + last_n_prices, function (data) {
      data.sort(
        (a:{price_close:string}, b:{price_close:string}) =>
          new Date(a.price_close).getTime()/1000 - new Date(b.price_close).getTime()/1000
      );
      scaleBreak = data[0].price_close * scale_factor;
      
      // scale_factor = 10
      for(var datum of data){
        dataPoints.push({
          x: new Date(datum["created_at"]).getTime()/1000,//datum["created_at"],
          y: datum["price_close"] * scale_factor
        });
      }
      
      dpsLength = dataPoints.length;
      chart.render();
      updateChart();
    });

    
    function updateChart() {
      console.log(dataPoints.length)
      console.log(dataPoints)

      $.getJSON("http://localhost:1234/services/last/" + last_n_prices, function (data) {
        data.sort(
          (a:{price_close:string}, b:{price_close:string}) =>
            new Date(a.price_close).getTime()/1000 - new Date(b.price_close).getTime()/1000
        );
        scaleBreak = data[0].price_close * scale_factor;
        console.log(scaleBreak)
        for(var datum of data){
          dataPoints.push({
            x: new Date(datum["created_at"]).getTime()/1000, //x: datum["created_at"],
            y: datum["price_close"] * scale_factor
          });
          dataPoints.shift()
        }

        // if (dataPoints.length > last_n_prices) {
        //   dataPoints = dataPoints.slice(-last_n_prices-1, last_n_prices-2);
        // }
        chart.render();
        setTimeout(function () { updateChart() }, 1000);
      });
    }
  }
}
