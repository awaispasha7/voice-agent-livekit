import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './app.component';
import { CallQueueComponent } from './components/call-queue/call-queue.component';
import { ActiveCallComponent } from './components/active-call/active-call.component';

import { LiveKitService } from './services/livekit.service';
import { Alive5SocketService } from './services/alive5-socket.service';
import { CallStateService } from './services/call-state.service';

@NgModule({
  declarations: [
    AppComponent,
    CallQueueComponent,
    ActiveCallComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [
    LiveKitService,
    Alive5SocketService,
    CallStateService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }

