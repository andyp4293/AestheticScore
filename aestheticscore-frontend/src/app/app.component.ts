import { Component } from '@angular/core';
import { FaceUploadComponent } from './components/face-upload/face-upload.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true,  
  imports: [
    FaceUploadComponent
  ]
})
export class AppComponent {
  title = 'AestheticScore';
}