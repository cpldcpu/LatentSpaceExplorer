import { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { InferenceSession, env, Tensor } from 'onnxruntime-web';
import msgpack from 'msgpack-lite';

// Add type declaration for msgpack-lite
declare module 'msgpack-lite';

// Type definitions
interface Metadata {
  x_coords: number[];
  y_coords: number[];
  density_maps: number[][][];
}

type DigitColors = {
  [key: number]: string;
};

const VAEExplorer = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const metadataInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 195, y: 195 }); // Center point adjusted for new size
  const [modelLoaded, setModelLoaded] = useState(false);
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  
  // Cyberpunk color palette for digits
  const digitColors: DigitColors = {
    0: '#FF4D4D',    // Neon Red
    1: '#00C6FF',    // Cyan
    2: '#FF00A8',    // Hot Pink
    3: '#01FFC3',    // Mint
    4: '#7700FF',    // Purple
    5: '#FF9B00',    // Orange
    6: '#00FFD1',    // Turquoise
    7: '#FF007B',    // Pink
    8: '#4DEEEA',    // Light Blue
    9: '#FFB800'     // Gold
  };

  useEffect(() => {
    env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';
    env.wasm.numThreads = 1;
    env.wasm.simd = true;

    // Load default ONNX model and metadata when component mounts
    loadDefaultModel();
    loadDefaultMetadata();

    console.log('ONNX Runtime environment configured');
  }, []);

  // Update session options type
  const sessionOptions: InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all' as const,
    enableCpuMemArena: false,
    enableMemPattern: false,
  };

  // Function to load default ONNX model
  const loadDefaultModel = async () => {
    try {
      setIsLoading(true);

      const modelUrl = 'model.onnx'; // Path to default model in public folder
      const response = await fetch(modelUrl);
      const modelBuffer = await response.arrayBuffer();

      // Convert ArrayBuffer to Uint8Array
      const modelData = new Uint8Array(modelBuffer);
      const session = await InferenceSession.create(modelData, sessionOptions);
      setSession(session);
      setModelLoaded(true);
    } catch (error) {
      console.error('Error loading default model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to load default metadata
  const loadDefaultMetadata = async () => {
    try {
      const metadataUrl = 'metadata.msgpack'; // Path to default metadata in public folder
      const response = await fetch(metadataUrl);
      const arrayBuffer = await response.arrayBuffer();
      const data = msgpack.decode(new Uint8Array(arrayBuffer)) as Metadata;
      setMetadata(data);
      drawLatentSpace(data);
    } catch (error) {
      console.error('Error loading default metadata:', error);
    }
  };

  const handleModelFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsLoading(true);
      try {
        // Create blob URL
        const blobUrl = URL.createObjectURL(file);
        
        // Load model using fetch to ensure proper streaming
        const response = await fetch(blobUrl);
        const modelBuffer = await response.arrayBuffer();
        
        // Revoke blob URL
        URL.revokeObjectURL(blobUrl);

        // Convert ArrayBuffer to Uint8Array
        const modelData = new Uint8Array(modelBuffer);
        const session = await InferenceSession.create(modelData, sessionOptions);
        setSession(session);
        setModelLoaded(true);
      } catch (error) {
        console.error('Error loading model:', error);
        alert('Failed to load model. Please ensure the file is a valid ONNX model.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleMetadataFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const data = msgpack.decode(new Uint8Array(arrayBuffer)) as Metadata;
        setMetadata(data);
        drawLatentSpace(data);
      } catch (error) {
        console.error('Error loading metadata:', error);
      }
    }
  };

  const drawLatentSpace = (metadata: Metadata) => {
    const canvas = mapCanvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Create a dark background
    ctx.fillStyle = '#0B1437';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid lines
    ctx.strokeStyle = '#1a2b5e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= width; i += 30) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    // Draw density maps
    const densityMaps = metadata.density_maps;
    const imageData = ctx.createImageData(width, height);
    
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        // Convert canvas coordinates to latent space coordinates
        const latentX = (x / width) * 6 - 3;
        const latentY = (y / height) * 6 - 3;
        
        // Find the nearest point in the metadata grid
        const xIdx = Math.floor((latentX + 3) * (metadata.x_coords.length - 1) / 6);
        const yIdx = Math.floor((latentY + 3) * (metadata.y_coords.length - 1) / 6);
        
        // Find the dominant class at this point
        let maxDensity = 0;
        let dominantClass = 0;
        
        for (let digit = 0; digit < 10; digit++) {
          const density = densityMaps[digit][yIdx][xIdx];
          if (density > maxDensity) {
            maxDensity = density;
            dominantClass = digit;
          }
        }

        // Get color for the dominant class
        const color = digitColors[dominantClass];
        const idx = (y * width + x) * 4;
        
        // Convert hex color to RGB
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        
        // Apply color with alpha based on density
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = Math.floor(maxDensity * 255 * 0.7); // 70% max opacity
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Add cyberpunk glow effect
    ctx.globalCompositeOperation = 'screen';
    ctx.filter = 'blur(4px)';
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'source-over';
  };

  const generateDigit = async (x: number, y: number) => {
    if (!session || !canvasRef.current) return;

    try {
      // Create tensor using Tensor
      const tensor = new Tensor('float32', new Float32Array([x, y]), [1, 2]);

      // Run inference
      const outputs = await session.run({
        'latent_vector': tensor
      });

      // Get output data
      const outputData = outputs['generated_image'].data as Float32Array;
      
      // Create a temporary canvas for the original 28x28 image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 28;
      tempCanvas.height = 28;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;
      
      // Draw the digit at original size
      const imageData = tempCtx.createImageData(28, 28);
      for (let i = 0; i < outputData.length; i++) {
        const idx = i * 4;
        const value = Math.floor(outputData[i] * 255);
        imageData.data[idx] = value;     // R
        imageData.data[idx + 1] = value; // G
        imageData.data[idx + 2] = value; // B
        imageData.data[idx + 3] = 255;   // A
      }
      tempCtx.putImageData(imageData, 0, 0);
      
      // Get the target canvas and clear it
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.fillStyle = '#0B143780'; // Changed from '#FB1437' to match the main background color
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Scale up the image using drawImage for better quality
      ctx.imageSmoothingEnabled = false; // Keep pixelated look
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
      
      // Add cyberpunk glow effect
      ctx.globalCompositeOperation = 'screen';
      ctx.shadowColor = '#00F6FFC0';
      ctx.shadowBlur = 10;
      ctx.drawImage(canvas, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      
      // Add scanlines
      for (let y = 0; y < canvas.height; y += 4) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, y, canvas.width, 2);
      }
    } catch (error) {
      console.error('Error generating digit:', error);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(!isDragging);
    const rect = e.currentTarget.getBoundingClientRect();
    const newPosition = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    setPosition(newPosition);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      const rect = e.currentTarget.getBoundingClientRect();
      const newPosition = {
        x: Math.min(Math.max(0, e.clientX - rect.left), 350), // Updated boundary
        y: Math.min(Math.max(0, e.clientY - rect.top), 350)   // Updated boundary
      };
      setPosition(newPosition);
    }
  };

  const handleMouseUp = () => {
    // setIsDragging(false);
  };

  // Update digit when position changes
  useEffect(() => {
    if (session) {
      const normalizedX = (position.x / 350) * 6 - 3; // Updated normalization
      const normalizedY = (position.y / 350) * 6 - 3; // Updated normalization
      generateDigit(normalizedX, normalizedY);
    }
  }, [position, session]);

  return (
    <Card className="h-screen w-screen bg-[#0B1437] text-cyan-400 border-cyan-500 rounded-none flex flex-col">
      <CardHeader className="flex flex-row justify-between items-center sticky top-0 z-10 bg-[#0B1437] border-b border-cyan-900 w-full">
        <CardTitle className="text-2xl font-mono tracking-wider">
          MNIST.VAE//:_EXPLORER  
        </CardTitle>
        <div className="flex items-center gap-4">
        <a 
              href="https://github.com/cpldcpu/LatentSpaceExplorer"   
            
            target="_blank" 
            rel="noopener noreferrer" 
           
          >
            <img 
              src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
              className="w-8 h-8"
            /></a>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleModelFileSelect}
            className="hidden"
            accept=".onnx"
          />
          <input
            type="file"
            ref={metadataInputRef}
            onChange={handleMetadataFileSelect}
            className="hidden"
            accept=".msgpack"
          />
          <Button 
            onClick={() => metadataInputRef.current?.click()}
            className="bg-orange-900 hover:bg-orange-800 text-orange-100 border border-orange-500"
          >
            <Upload className="mr-2 h-4 w-4" />
            {metadata ? 'Metadata Loaded' : 'Load Metadata'}
          </Button>
          <Button 
            onClick={() => fileInputRef.current?.click()}
            className={`bg-cyan-900 hover:bg-cyan-800 text-cyan-100 border border-cyan-500 ${
              isLoading ? 'opacity-50 cursor-wait' : ''
            }`}
            disabled={isLoading}
          >
            <Upload className="mr-2 h-4 w-4" />
            {isLoading ? 'Loading...' : modelLoaded ? 'Model Loaded' : 'Load VAE Model'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-4 flex flex-col items-center justify-center">
        <div className="flex flex-col md:flex-row gap-6 items-center justify-center">
          <div className="relative">
            <div className="border-2 border-cyan-500 p-2 bg-[#0B1437]">
              <canvas
                ref={canvasRef}
                width={350}
                height={350}
                className="border border-cyan-800"
              />
            </div>
            <p className="text-center mt-2 font-mono">GENERATED::OUTPUT</p>
          </div>          
          <div className="relative">
            <div className="border-2 border-orange-500 p-2 bg-[#0B1437]">
              <canvas
                ref={mapCanvasRef}
                width={350}
                height={350}
                className="border border-orange-800 cursor-pointer"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
              <div 
                className="absolute w-6 h-6 border-2 border-cyan-400 rounded-full mt-3 ml-3"
                style={{
                  left: position.x,
                  top: position.y,
                  cursor: isDragging ? 'grabbing' : 'grab',
                  boxShadow: '0 0 10px #00F6FF, 0 0 20px #00F6FF',
                }}
              />
            </div>
            <p className="text-center mt-2 font-mono">LATENT::SPACE::NAVIGATOR</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 justify-center font-mono text-xs mt-2"> 
          {Object.entries(digitColors).map(([digit, color]) => (
            <div key={digit} 
                 className="flex items-center gap-1 bg-[#162555] px-2 py-1 rounded border border-cyan-900"
                 style={{ minWidth: '60px' }}>
              <div 
                className="w-3 h-3 rounded-sm"
                style={{ 
                  backgroundColor: color,
                  boxShadow: `0 0 5px ${color}`
                }}
              />
              <span className="text-lg pl-2">{digit}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default VAEExplorer;