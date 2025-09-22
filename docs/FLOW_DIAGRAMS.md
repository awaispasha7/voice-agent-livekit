# 🔄 Alive5 Voice Agent - Flow Diagrams

## System Architecture Flow

```mermaid
graph TB
    A[User Voice Input] --> B[Frontend - LiveKit]
    B --> C[Worker - Voice Processing]
    C --> D[Backend API - Flow Logic]
    D --> E{Intent Detection}
    E -->|Escalation| F[Agent Transfer]
    E -->|LLM Analysis| G[Flow Processing]
    G --> H[Response Generation]
    H --> I[Worker - Voice Output]
    I --> J[Frontend - Audio Playback]
    F --> K[Human Agent]
    
    style A fill:#e1f5fe
    style K fill:#f3e5f5
    style E fill:#fff3e0
    style G fill:#e8f5e8
```

## Intent Detection Flow

```mermaid
graph TD
    A[User Message] --> B{Pattern Match Check}
    B -->|Agent Keywords| C[Transfer Initiated]
    B -->|No Match| D[LLM Intent Detection]
    D --> E{Intent Found?}
    E -->|Yes| F[Flow Processing]
    E -->|No| G[FAQ Bot Fallback]
    F --> H[Response Generation]
    G --> H
    C --> I[Human Agent Connection]
    
    style A fill:#e1f5fe
    style C fill:#ffebee
    style F fill:#e8f5e8
    style G fill:#fff3e0
    style I fill:#f3e5f5
```

## Conversation Flow Examples

### 1. Agent Transfer Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant W as Worker
    participant B as Backend
    participant A as Agent
    
    U->>F: "Can I speak with someone?"
    F->>W: Voice Data
    W->>B: Process Message
    B->>B: Escalation Detection
    B->>W: Transfer Initiated
    W->>F: "Connecting to agent..."
    F->>U: Audio Response
    B->>A: Transfer Request
    A->>U: Human Agent Connected
```

### 2. Pricing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant W as Worker
    participant B as Backend
    
    U->>F: "I need pricing"
    F->>W: Voice Data
    W->>B: Process Message
    B->>B: LLM Intent Detection
    B->>W: Flow Started (Pricing)
    W->>F: "How many phone lines?"
    F->>U: Audio Response
    
    U->>F: "5 lines"
    F->>W: Voice Data
    W->>B: Process Response
    B->>W: Next Question
    W->>F: "How many texts per month?"
    F->>U: Audio Response
    
    U->>F: "2000 texts"
    F->>W: Voice Data
    W->>B: Process Response
    B->>W: Final Question
    W->>F: "Any special needs?"
    F->>U: Audio Response
    
    U->>F: "Salesforce integration"
    F->>W: Voice Data
    W->>B: Process Response
    B->>W: Flow Complete
    W->>F: "Generating plan..."
    F->>U: Audio Response
```

### 3. Weather Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant W as Worker
    participant B as Backend
    participant API as Weather API
    
    U->>F: "What's the weather?"
    F->>W: Voice Data
    W->>B: Process Message
    B->>B: LLM Intent Detection
    B->>W: Flow Started (Weather)
    W->>F: "What's your zip code?"
    F->>U: Audio Response
    
    U->>F: "90210"
    F->>W: Voice Data
    W->>B: Process Response
    B->>API: Get Weather Data
    API->>B: Weather Information
    B->>W: Weather Response
    W->>F: "Current weather in Beverly Hills..."
    F->>U: Audio Response
```

## State Management Flow

```mermaid
stateDiagram-v2
    [*] --> NewSession
    NewSession --> IntentDetection
    IntentDetection --> GreetingFlow : Greeting Intent
    IntentDetection --> PricingFlow : Pricing Intent
    IntentDetection --> WeatherFlow : Weather Intent
    IntentDetection --> AgentTransfer : Agent Intent
    IntentDetection --> FAQFallback : No Intent
    
    GreetingFlow --> ConversationActive
    PricingFlow --> ConversationActive
    WeatherFlow --> ConversationActive
    FAQFallback --> ConversationActive
    
    ConversationActive --> FlowProcessing
    FlowProcessing --> ConversationActive : Continue Flow
    FlowProcessing --> FlowComplete : Flow Finished
    FlowProcessing --> AgentTransfer : Escalation
    
    AgentTransfer --> HumanAgent
    FlowComplete --> SessionEnd
    HumanAgent --> SessionEnd
    SessionEnd --> [*]
    
    note right of AgentTransfer
        Immediate response:
        "Connecting to agent..."
    end note
```

## Error Handling Flow

```mermaid
graph TD
    A[Request Received] --> B{Valid Request?}
    B -->|No| C[400 Bad Request]
    B -->|Yes| D[Process Request]
    D --> E{Processing Success?}
    E -->|No| F{Error Type}
    F -->|Network| G[Retry Logic]
    F -->|Validation| H[400 Bad Request]
    F -->|Server| I[500 Internal Error]
    E -->|Yes| J[Success Response]
    G --> K{Retry Success?}
    K -->|Yes| J
    K -->|No| I
    
    style C fill:#ffebee
    style H fill:#ffebee
    style I fill:#ffebee
    style J fill:#e8f5e8
```

## Deployment Flow

```mermaid
graph LR
    A[Code Repository] --> B[GitHub]
    B --> C[Vercel - Frontend]
    B --> D[Render - Backend]
    B --> E[Render - Background Worker]
    
    C --> F[Frontend Deployed]
    D --> G[Backend Deployed]
    E --> H[Worker Deployed]
    
    F --> I[Production System]
    G --> I
    H --> I
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Voice Input] --> B[LiveKit Client]
        B --> C[Audio Processing]
        C --> D[Real-time Communication]
    end
    
    subgraph "Worker Layer"
        E[Voice Processing] --> F[LiveKit Agent]
        F --> G[Conversation Management]
        G --> H[Backend Communication]
    end
    
    subgraph "Backend Layer"
        I[API Endpoints] --> J[Flow Logic]
        J --> K[Intent Detection]
        K --> L[State Management]
        L --> M[Response Generation]
    end
    
    subgraph "External Services"
        N[OpenAI API] --> K
        O[Alive5 API] --> J
        P[Weather API] --> J
    end
    
    D --> E
    H --> I
    M --> H
    
    style A fill:#e1f5fe
    style N fill:#fff3e0
    style O fill:#fff3e0
    style P fill:#fff3e0
```

## Performance Monitoring Flow

```mermaid
graph TD
    A[Request] --> B[Load Balancer]
    B --> C[API Gateway]
    C --> D[Rate Limiting]
    D --> E[Authentication]
    E --> F[Request Processing]
    F --> G[Response Generation]
    G --> H[Logging]
    H --> I[Metrics Collection]
    I --> J[Monitoring Dashboard]
    
    F --> K{Error?}
    K -->|Yes| L[Error Logging]
    K -->|No| M[Success Logging]
    L --> H
    M --> H
    
    style A fill:#e1f5fe
    style J fill:#e8f5e8
    style L fill:#ffebee
    style M fill:#e8f5e8
```

## Security Flow

```mermaid
graph TD
    A[Incoming Request] --> B[HTTPS Validation]
    B --> C[Rate Limiting]
    C --> D[JWT Validation]
    D --> E{Valid Token?}
    E -->|No| F[401 Unauthorized]
    E -->|Yes| G[Request Processing]
    G --> H[Data Validation]
    H --> I{Valid Data?}
    I -->|No| J[400 Bad Request]
    I -->|Yes| K[Process Request]
    K --> L[Response Generation]
    L --> M[Secure Response]
    
    style A fill:#e1f5fe
    style F fill:#ffebee
    style J fill:#ffebee
    style M fill:#e8f5e8
```

## Session Lifecycle

```mermaid
graph TD
    A[Session Start] --> B[Connection Established]
    B --> C[Initial Greeting]
    C --> D[User Interaction]
    D --> E{Intent Detected?}
    E -->|Yes| F[Flow Processing]
    E -->|No| G[FAQ Response]
    F --> H{Flow Complete?}
    H -->|No| D
    H -->|Yes| I[Session Summary]
    G --> J{Continue?}
    J -->|Yes| D
    J -->|No| I
    I --> K[Session Cleanup]
    K --> L[Session End]
    
    style A fill:#e1f5fe
    style L fill:#f3e5f5
    style F fill:#e8f5e8
    style G fill:#fff3e0
```

---

These diagrams provide a comprehensive visual representation of how the Alive5 Voice Agent system works, from high-level architecture to detailed conversation flows. They can be used for client presentations, technical documentation, and system understanding.
