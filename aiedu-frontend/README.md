# 🚀 AIEDU Frontend - Next.js with TypeScript

Modern React frontend for the AIEDU RAG system with hybrid provider management and real-time speech interaction.

## 🏗️ Architecture

- **Framework**: Next.js 15.3.3 with App Router
- **Language**: TypeScript with strict type checking
- **Styling**: Tailwind CSS 3.x for utility-first design
- **Icons**: Lucide React for consistent iconography
- **HTTP Client**: Axios with request/response interceptors
- **State Management**: React hooks with custom providers
- **Build Tool**: Turbopack for fast development builds

## 📁 Project Structure

```
src/
├── app/                         # Next.js App Router
│   ├── layout.tsx               # Root layout with global styles
│   ├── page.tsx                 # Homepage with provider demo
│   └── globals.css              # Global CSS and Tailwind imports
├── components/
│   ├── providers/               # Provider management components
│   │   ├── ProviderSelector.tsx # Local/Cloud toggle
│   │   ├── StatusIndicator.tsx  # Service health display
│   │   └── ProviderPanel.tsx    # Complete sidebar panel
│   ├── chat/                    # Chat interface (Week 2)
│   ├── speech/                  # Speech components (Week 2)
│   └── knowledge/               # Knowledge base management (Week 2)
├── hooks/
│   ├── useProvider.ts           # Provider state management
│   ├── useSpeech.ts             # Speech recording/playback (Week 2)
│   └── useRAG.ts                # RAG query handling (Week 2)
├── services/
│   ├── api.ts                   # Centralized API client
│   ├── speechService.ts         # Speech-specific API calls (Week 2)
│   └── ragService.ts            # RAG-specific API calls (Week 2)
├── types/
│   ├── provider.ts              # Provider interfaces
│   ├── speech.ts                # Speech interfaces
│   └── rag.ts                   # RAG interfaces
└── utils/
    ├── audioUtils.ts            # Audio processing utilities (Week 2)
    └── constants.ts             # Application constants
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ with npm
- Flask backend running at `https://127.0.0.1:5000`

### Installation & Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

**Access**: http://localhost:3000

### Build & Production

```bash
# Build for production
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint
```

## 🔧 Configuration

### Environment Variables

Create `.env.local` file:

```bash
# Backend API configuration
NEXT_PUBLIC_API_URL=https://127.0.0.1:5000

# Development settings
NEXT_PUBLIC_ENV=development
```

### TypeScript Configuration

The project uses strict TypeScript configuration:

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true
  }
}
```

## 🎯 Key Features

### 🎛️ Provider Management System

Real-time switching between local and cloud AI providers:

```typescript
import { useProvider } from '@/hooks/useProvider'

const { currentProvider, status, switchProvider, loading } = useProvider()

// Switch to cloud providers
await switchProvider('cloud')

// Check service status
console.log(status) // { stt: 'working', tts: 'working', llm: 'working' }
```

### 📡 API Integration

Centralized API client with full type safety:

```typescript
import { apiClient } from '@/services/api'

// Get current providers
const providers = await apiClient.getCurrentProviders()

// Force provider switch
await apiClient.forceProvider('local')

// Synthesize speech
const audio = await apiClient.synthesize({
  text: "Hello world",
  voice: "en-US",
  language: "en"
})
```

### 🎨 Component Architecture

Modular React components with TypeScript:

```typescript
// Provider status indicator
<StatusIndicator 
  service="stt" 
  status={status.stt} 
  className="w-4 h-4" 
/>

// Provider selector
<ProviderSelector
  currentProvider={currentProvider}
  onProviderChange={switchProvider}
  disabled={loading}
/>

// Complete provider panel
<ProviderPanel />
```

## 🔄 Real-Time Development

The frontend provides instant hot reload for all changes:

- **React Components**: Instant updates with state preservation
- **TypeScript Files**: Real-time type checking and IntelliSense
- **CSS/Tailwind**: Immediate style updates
- **API Services**: Automatic reconnection to backend

### Development Workflow

1. **Edit Components**: Changes appear instantly in browser
2. **TypeScript Safety**: Compile-time error checking
3. **API Integration**: Real-time backend communication
4. **State Management**: Persistent state across hot reloads

## 📱 Responsive Design

Built with mobile-first approach using Tailwind CSS:

- **Desktop**: 75% main content + 25% provider sidebar
- **Tablet**: Responsive grid layout
- **Mobile**: Stacked components with collapsible sidebar

### Layout Components

```typescript
// Main layout with sidebar
<div className="min-h-screen bg-gray-50">
  <div className="container mx-auto px-4 py-8">
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
      {/* Main content - 75% */}
      <main className="lg:col-span-3">
        {children}
      </main>
      
      {/* Provider sidebar - 25% */}
      <aside className="lg:col-span-1">
        <ProviderPanel />
      </aside>
    </div>
  </div>
</div>
```

## 🎨 UI Components

### Design System

- **Colors**: Tailwind color palette with semantic usage
- **Typography**: Inter font with responsive scaling
- **Icons**: Lucide React with consistent sizing
- **Buttons**: Interactive states with loading indicators
- **Forms**: Accessible inputs with validation

### Status Indicators

```typescript
// Service health colors
const statusColors = {
  working: 'text-green-500',
  connecting: 'text-yellow-500', 
  error: 'text-red-500',
  unknown: 'text-gray-400'
}

// Status badges
<span className={`inline-flex items-center ${statusColors[status]}`}>
  {getStatusIcon(status)}
  {getStatusText(status)}
</span>
```

## 📊 State Management

### Provider State Hook

```typescript
export const useProvider = () => {
  const [currentProvider, setCurrentProvider] = useState<ProviderType>('local')
  const [status, setStatus] = useState<ProviderStatus>({
    stt: 'unknown',
    tts: 'unknown', 
    llm: 'unknown'
  })
  const [loading, setLoading] = useState(false)

  // Real-time status polling (5 seconds)
  useEffect(() => {
    const interval = setInterval(refreshStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  return { currentProvider, status, switchProvider, loading }
}
```

### API Error Handling

```typescript
// Request interceptor for logging
apiClient.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)
```

## 🧪 Testing Strategy

### Component Testing
- React Testing Library for component tests
- Jest for unit testing
- MSW for API mocking

### Type Safety
- Strict TypeScript configuration
- Interface validation for API responses
- Compile-time error prevention

### Integration Testing
- E2E testing with provider switching
- API integration testing
- Real-time status monitoring tests

## 📦 Dependencies

### Core Dependencies
```json
{
  "next": "15.3.3",
  "react": "19.0.0",
  "typescript": "5.x",
  "tailwindcss": "3.x",
  "axios": "^1.7.2",
  "lucide-react": "^0.451.0"
}
```

### Development Dependencies
```json
{
  "@types/node": "^20",
  "@types/react": "^19",
  "eslint": "^8",
  "eslint-config-next": "15.3.3"
}
```

## 🚀 Deployment

### Production Build
```bash
# Build optimized production bundle
npm run build

# Start production server
npm start
```

### Vercel Deployment (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel --prod
```

### Environment Configuration
```bash
# Production environment variables
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
NEXT_PUBLIC_ENV=production
```

## 🔗 Integration with Backend

### CORS Configuration
The backend Flask app is configured for cross-origin requests:

```python
# Backend CORS setup (already configured)
CORS(app, origins=['http://localhost:3000'], 
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
```

### API Communication
- **Frontend**: http://localhost:3000
- **Backend**: https://127.0.0.1:5000
- **Communication**: Axios with HTTPS support and proper headers

## 📝 Development Guidelines

### Code Style
- Use TypeScript for all new files
- Follow React hooks patterns
- Implement proper error boundaries
- Use Tailwind for all styling

### Component Patterns
- Export components as default
- Use TypeScript interfaces for props
- Implement proper loading states
- Handle errors gracefully

### API Integration
- Use the centralized API client
- Implement proper type safety
- Handle loading and error states
- Use React Query for complex data fetching (Week 2)

## 🎯 Roadmap

### Phase 3 Week 2 - Feature Migration
- [ ] Chat interface with React components
- [ ] Speech recording with MediaRecorder API
- [ ] RAG query processing with React state
- [ ] Audio playback controls with Web Audio API
- [ ] Document management interface
- [ ] Advanced error handling and retry logic

### Future Enhancements
- [ ] React Query for advanced data fetching
- [ ] Zustand for global state management
- [ ] React Hook Form for form handling
- [ ] Framer Motion for animations
- [ ] PWA capabilities for offline usage

---

**Modern React Frontend for Educational AI with TypeScript Safety** ⚛️
