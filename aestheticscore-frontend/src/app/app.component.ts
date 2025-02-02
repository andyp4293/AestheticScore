import { Component } from '@angular/core';
import { FaceUploadComponent } from './components/face-upload/face-upload.component';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  standalone: true,  
  imports: [
    FaceUploadComponent,
    HttpClientModule
  ]
})
export class AppComponent {
  title = 'AestheticScore';
}