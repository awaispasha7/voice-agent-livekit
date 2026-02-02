import { Component, OnInit, OnDestroy } from '@angular/core';
import { Alive5SocketService } from './services/alive5-socket.service';
import { AgentProfileService } from './services/agent-profile.service';
import { environment } from '../environments/environment';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Alive5 HITL Dashboard';

  constructor(
    private alive5Socket: Alive5SocketService,
    private agentProfile: AgentProfileService
  ) {}

  ngOnInit(): void {
    console.log('[App] Initializing dashboard...');
    
    // Prompt for agent name if not set
    if (!this.agentProfile.hasProfile()) {
      this.promptForAgentName();
    }
    
    // Get agent ID from profile or generate a temporary one
    const profile = this.agentProfile.getProfile();
    const agentId = profile?.agentId || `dashboard_${Date.now()}`;
    
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

  /**
   * Prompt user to enter their name
   */
  private promptForAgentName(): void {
    const name = prompt('Welcome to Alive5 HITL Dashboard!\n\nPlease enter your name:');
    
    if (name && name.trim()) {
      this.agentProfile.setProfile(name.trim());
      console.log('[App] Agent profile created:', name);
    } else {
      // If no name provided, prompt again after a short delay
      setTimeout(() => {
        if (!this.agentProfile.hasProfile()) {
          this.promptForAgentName();
        }
      }, 1000);
    }
  }
}

