import { Component, OnInit, OnDestroy } from '@angular/core';
import { Alive5SocketService } from './services/alive5-socket.service';
import { environment } from '../environments/environment';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Alive5 HITL Dashboard';

  constructor(private alive5Socket: Alive5SocketService) {}

  ngOnInit(): void {
    console.log('[App] Initializing Alive5 Socket connection...');
    
    // Generate a unique agent ID for this dashboard instance
    const agentId = `dashboard_${Date.now()}`;
    
    // Connect to Alive5 Socket.IO
    try {
      this.alive5Socket.connect(
        environment.alive5ApiKey,
        agentId,
        environment.alive5SocketUrl
      );
      console.log('[App] Socket connection initiated');
    } catch (error) {
      console.error('[App] Failed to initialize socket:', error);
    }
  }

  ngOnDestroy(): void {
    console.log('[App] Disconnecting socket...');
    this.alive5Socket.disconnect();
  }
}

