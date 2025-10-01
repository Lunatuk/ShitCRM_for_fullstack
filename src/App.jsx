import React, { useMemo, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  ChakraProvider,
  extendTheme,
  Box,
  Flex,
  HStack,
  VStack,
  Text,
  Heading,
  IconButton,
  Button,
  Avatar,
  Badge,
  Input,
  InputGroup,
  InputLeftElement,
  Textarea,
  Select,
  NumberInput,
  NumberInputField,
  useDisclosure,
  useToast,
  Drawer,
  DrawerOverlay,
  DrawerContent,
  DrawerHeader,
  DrawerBody,
  DrawerFooter,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Spacer,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Kbd,
  Tooltip,
  Switch,
  useColorMode,
  useColorModeValue,
  Tag,
  SimpleGrid
} from "@chakra-ui/react";
import { BrowserRouter, Routes, Route, Link, useNavigate, useLocation } from "react-router-dom";
import { Search, Menu as MenuIcon, LogOut, Plus, Upload, CheckCircle2, AlertCircle, FileText, Wallet, Plane, PieChart, Settings, Camera, SunMedium, MoonStar, ChevronRight } from "lucide-react";

/**
 * FRONTEND-ONLY MVP (React + Chakra UI)
 * - Pure client-side demo with mock data and API placeholders
 * - Pages: Login, Dashboard, Travel Requests (list + new), Expenses, Reports, Budgets, Admin
 * - Components: AppShell, Sidebar, Topbar, Tables, Forms, Modals
 * - "OCR" button is stubbed (simulates extraction)
 */

// ---- THEME ----
const theme = extendTheme({
  fonts: {
    heading: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'",
    body: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'",
  },
  components: {
    Button: { baseStyle: { borderRadius: "xl" } },
    Input: { baseStyle: { borderRadius: "lg" } },
  },
});

// ---- MOCKS & UTIL ----
const mockUser = { id: 1, name: "Ferdin", email: "ferdin@example.com", role: "Employee", dept: "Production" };

const projects = [
  { id: 1, name: "Film Festival Tour", dept: "Production", currency: "EUR" },
  { id: 2, name: "Analytics Rollout", dept: "IT", currency: "EUR" },
];

const categories = [
  { id: 1, name: "Hotel" },
  { id: 2, name: "Flights" },
  { id: 3, name: "Meals" },
  { id: 4, name: "Taxi" },
  { id: 5, name: "Other" },
];

let idCounter = 1000;
const nextId = () => ++idCounter;

// simple in-memory stores
const store = {
  travelRequests: [
    { id: 101, title: "Berlin → Tbilisi (conference)", projectId: 1, employee: mockUser.name, startDate: "2025-10-10", endDate: "2025-10-15", perDiem: 60, needAdvance: true, status: "Approved", estimate: 850 },
    { id: 102, title: "Rome scouting", projectId: 1, employee: mockUser.name, startDate: "2025-11-05", endDate: "2025-11-08", perDiem: 50, needAdvance: false, status: "Submitted", estimate: 420 },
  ],
  expenses: [
    { id: 301, requestId: 101, employeeId: 1, categoryId: 3, amount: 24.90, currency: "EUR", spendAt: "2025-10-11", merchant: "Cafe 123", note: "Lunch", fileName: "receipt_301.jpg", ocr: { amount: 24.90, date: "2025-10-11", merchant: "Cafe 123" }, verified: true },
  ],
  reports: [
    { id: 501, requestId: 101, employeeId: 1, status: "Closed", total: 24.90, submittedAt: "2025-10-16" },
  ],
  budgets: [
    { id: 801, scope: { projectId: 1 }, period: "2025-Q4", currency: "EUR", amount: 10000, committed: 1270, actual: 900 },
    { id: 802, scope: { projectId: 2 }, period: "2025-Q4", currency: "EUR", amount: 4000, committed: 0, actual: 0 },
  ]
};

// pretend API
const api = {
  login: async (email, password) => {
    await wait(400);
    return { token: "demo-token", user: mockUser };
  },
  getTravelRequests: async () => { await wait(200); return [...store.travelRequests]; },
  createTravelRequest: async (payload) => { await wait(350); const item = { id: nextId(), status: "Draft", estimate: 0, ...payload }; store.travelRequests.unshift(item); return item; },
  submitTravelRequest: async (id) => { await wait(250); const r = store.travelRequests.find(x => x.id===id); if (r) r.status = "Submitted"; return r; },
  getExpensesByRequest: async (requestId) => { await wait(200); return store.expenses.filter(e=>e.requestId===requestId); },
  createExpense: async (payload) => { await wait(300); const item = { id: nextId(), verified: false, ...payload }; store.expenses.unshift(item); return item; },
  getReports: async () => { await wait(200); return [...store.reports]; },
  createReportFromRequest: async (requestId) => { await wait(400); const total = store.expenses.filter(e=>e.requestId===requestId).reduce((s,e)=>s+e.amount,0); const rep = { id: nextId(), requestId, employeeId: 1, status: "Submitted", total, submittedAt: new Date().toISOString().slice(0,10) }; store.reports.unshift(rep); return rep; },
  getBudgets: async () => { await wait(200); return [...store.budgets]; },
  // fake OCR
  ocrExtract: async (file) => { await wait(800); return { amount: +(Math.random()*50+5).toFixed(2), date: new Date().toISOString().slice(0,10), merchant: "Demo Merchant", confidence: +(Math.random()*0.4+0.6).toFixed(2) }; }
};

function wait(ms){ return new Promise(res=>setTimeout(res, ms)); }

// ---- APP SHELL ----
function ColorModeToggle(){
  const { colorMode, toggleColorMode } = useColorMode();
  return (
    <Tooltip label={`Switch to ${colorMode==='light'?'dark':'light'} mode`}>
      <IconButton aria-label="toggle color mode" size="sm" variant="ghost" onClick={toggleColorMode} icon={colorMode==='light'? <MoonStar size={18}/> : <SunMedium size={18}/> } />
    </Tooltip>
  );
}

function Sidebar(){
  const nav = [
    { to: "/dashboard", label: "Dashboard", icon: PieChart },
    { to: "/travel", label: "Travel Requests", icon: Plane },
    { to: "/expenses", label: "Expenses", icon: Wallet },
    { to: "/reports", label: "Reports", icon: FileText },
    { to: "/budgets", label: "Budgets", icon: PieChart },
    { to: "/admin", label: "Admin", icon: Settings },
  ];
  const location = useLocation();
  const bg = useColorModeValue("white", "gray.800");
  const activeBg = useColorModeValue("gray.100", "gray.700");
  return (
    <VStack as="nav" spacing={1} align="stretch" p={3} w={{base:"full", md:64}} bg={bg} borderRightWidth={{md:1}}>
      <HStack px={2} py={3}>
        <Badge colorScheme="purple" borderRadius="md" px={2}>MVP</Badge>
        <Heading size="sm">T&E Portal</Heading>
      </HStack>
      {nav.map((item)=>{
        const Icon = item.icon;
        const active = location.pathname.startsWith(item.to);
        return (
          <Flex as={Link} key={item.to} to={item.to} align="center" gap={3} px={3} py={2} borderRadius="lg" _hover={{bg:activeBg}} bg={active?activeBg:undefined}>
            <Icon size={18} />
            <Text>{item.label}</Text>
            <Spacer/>
            {active && <ChevronRight size={16}/>}    
          </Flex>
        );
      })}
      <Spacer/>
      <Box px={3} py={2} fontSize="xs" color="gray.500">
        <Text>Logged as</Text>
        <HStack>
          <Avatar size="xs" name={mockUser.name}/>
          <Text>{mockUser.name}</Text>
          <Tag>{mockUser.role}</Tag>
        </HStack>
      </Box>
    </VStack>
  );
}

function Topbar(){
  const navigate = useNavigate();
  const barBg = useColorModeValue("white","gray.800");
  return (
    <HStack px={4} py={2} borderBottomWidth={1} bg={barBg}>
      <HStack display={{base:"flex", md:"none"}}>
        <Menu>
          <MenuButton as={IconButton} icon={<MenuIcon size={18}/>} variant="ghost"/>
          <MenuList>
            <MenuItem as={Link} to="/dashboard">Dashboard</MenuItem>
            <MenuItem as={Link} to="/travel">Travel Requests</MenuItem>
            <MenuItem as={Link} to="/expenses">Expenses</MenuItem>
            <MenuItem as={Link} to="/reports">Reports</MenuItem>
            <MenuItem as={Link} to="/budgets">Budgets</MenuItem>
            <MenuItem as={Link} to="/admin">Admin</MenuItem>
          </MenuList>
        </Menu>
        <Heading size="sm">T&E Portal</Heading>
      </HStack>
      <Spacer/>
      <InputGroup maxW={320} display={{base:"none", md:"block"}}>
        <InputLeftElement pointerEvents='none'>
          <Search size={16}/>
        </InputLeftElement>
        <Input placeholder="Quick search (mock)" borderRadius="xl"/>
      </InputGroup>
      <HStack>
        <ColorModeToggle/>
        <Tooltip label="Logout (demo)"><IconButton aria-label="logout" size="sm" variant="ghost" onClick={()=>navigate("/")} icon={<LogOut size={18}/>}/></Tooltip>
      </HStack>
    </HStack>
  );
}

function AppShell({children}){
  return (
    <Flex h="100dvh" overflow="hidden" bg={useColorModeValue("gray.50","gray.900")}> 
      <Box display={{base:"none", md:"block"}}><Sidebar/></Box>
      <Flex direction="column" flex={1} minW={0}>
        <Topbar/>
        <Box as="main" p={4} overflowY="auto">{children}</Box>
      </Flex>
    </Flex>
  );
}

// ---- PAGES ----
function LoginPage(){
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();

  async function onLogin(){
    setLoading(true);
    try{
      await api.login(email, password);
      toast({ title: "Welcome!", description: "Demo login successful", status: "success" });
      navigate("/dashboard");
    }catch(e){
      toast({ title: "Login failed", status: "error"});
    }finally{ setLoading(false); }
  }

  return (
    <Flex minH="100dvh" align="center" justify="center" bg={useColorModeValue("gray.50","gray.900")}> 
      <Box bg={useColorModeValue("white","gray.800")} p={8} borderRadius="2xl" boxShadow="lg" w="full" maxW="md">
        <VStack align="stretch" spacing={5}>
          <Heading size="md">Sign in to T&E Portal</Heading>
          <Input placeholder="Email" value={email} onChange={e=>setEmail(e.target.value)} />
          <Input placeholder="Password" type="password" value={password} onChange={e=>setPassword(e.target.value)} />
          <Button onClick={onLogin} isLoading={loading} colorScheme="purple">Sign in</Button>
          <Text fontSize="xs" color="gray.500">Use any credentials (demo)</Text>
        </VStack>
      </Box>
    </Flex>
  );
}

function DashboardPage(){
  const [reqs, setReqs] = useState([]);
  const [budgets, setBudgets] = useState([]);
  useEffect(()=>{ api.getTravelRequests().then(setReqs); api.getBudgets().then(setBudgets); },[]);

  const totalSubmitted = reqs.filter(r=>r.status!=="Draft").length;
  const totalEstimate = reqs.reduce((s,r)=>s+(r.estimate||0),0);

  return (
    <AppShell>
      <VStack align="stretch" spacing={6}>
        <Heading size="md">Overview</Heading>
        <SimpleGrid columns={{base:1, md:3}} spacing={4}>
          <StatCard label="Active Requests" value={totalSubmitted} help="submitted/approved"/>
          <StatCard label="Est. Spend (EUR)" value={totalEstimate.toFixed(2)} help="sum of estimates"/>
          <StatCard label="Projects" value={projects.length} help="tracked"/>
        </SimpleGrid>
        <Box>
          <Heading size="sm" mb={3}>Budgets snapshot</Heading>
          <SimpleGrid columns={{base:1, md:2}} spacing={4}>
            {budgets.map(b=> <BudgetCard key={b.id} budget={b} />)}
          </SimpleGrid>
        </Box>
        <Box>
          <Heading size="sm" mb={3}>Recent requests</Heading>
          <RequestsTable items={reqs.slice(0,5)} compact/>
        </Box>
      </VStack>
    </AppShell>
  );
}

function StatCard({label, value, help}){
  return (
    <Box p={4} bg={useColorModeValue("white","gray.800")} borderRadius="2xl" boxShadow="sm" borderWidth={1}>
      <Stat>
        <StatLabel>{label}</StatLabel>
        <StatNumber>{value}</StatNumber>
        {help && <StatHelpText>{help}</StatHelpText>}
      </Stat>
    </Box>
  );
}

function BudgetCard({budget}){
  const { amount, committed, actual, currency, period } = budget;
  const used = actual; const percent = Math.min(100, Math.round((used/amount)*100));
  const color = percent>85 ? "red" : percent>65 ? "orange" : "green";
  return (
    <Box p={4} bg={useColorModeValue("white","gray.800")} borderRadius="2xl" boxShadow="sm" borderWidth={1}>
      <HStack justify="space-between" mb={2}>
        <Heading size="sm">{period}</Heading>
        <Badge colorScheme={color}>{percent}% used</Badge>
      </HStack>
      <Text fontSize="sm" color="gray.500">{currency} {used.toFixed(2)} of {amount.toFixed(2)}</Text>
      <Progress mt={2} value={percent} colorScheme={color}/>
      <HStack mt={2} fontSize="xs" color="gray.500" spacing={6}>
        <HStack><Kbd>Committed</Kbd><Text>{currency} {committed.toFixed(2)}</Text></HStack>
        <HStack><Kbd>Actual</Kbd><Text>{currency} {actual.toFixed(2)}</Text></HStack>
      </HStack>
    </Box>
  );
}

function TravelRequestsPage(){
  const [items, setItems] = useState([]);
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [form, setForm] = useState({ title:"", projectId: projects[0].id, startDate:"", endDate:"", perDiem: 50, needAdvance: false });

  useEffect(()=>{ api.getTravelRequests().then(setItems); },[]);

  async function create(){
    const payload = { ...form, employee: mockUser.name };
    const created = await api.createTravelRequest(payload);
    toast({ title: "Draft created", status: "success"});
    setItems(prev=>[created, ...prev]);
    onClose();
  }

  async function submit(id){ await api.submitTravelRequest(id); setItems(prev=>prev.map(x=>x.id===id?{...x, status:"Submitted"}:x)); toast({ title:"Submitted", status:"success"}); }

  return (
    <AppShell>
      <HStack mb={4} justify="space-between">
        <Heading size="md">Travel Requests</Heading>
        <Button leftIcon={<Plus size={16}/>} colorScheme="purple" onClick={onOpen}>New request</Button>
      </HStack>
      <RequestsTable items={items} onSubmit={submit}/>

      <Drawer isOpen={isOpen} placement='right' onClose={onClose} size="md">
        <DrawerOverlay/> 
        <DrawerContent>
          <DrawerHeader>New travel request</DrawerHeader>
          <DrawerBody>
            <VStack align="stretch" spacing={4}>
              <FormRow label="Title"><Input value={form.title} onChange={e=>setForm({...form, title:e.target.value})}/></FormRow>
              <FormRow label="Project">
                <Select value={form.projectId} onChange={e=>setForm({...form, projectId:Number(e.target.value)})}>
                  {projects.map(p=> <option key={p.id} value={p.id}>{p.name}</option>)}
                </Select>
              </FormRow>
              <FormRow label="Dates">
                <HStack>
                  <Input type="date" value={form.startDate} onChange={e=>setForm({...form, startDate:e.target.value})}/>
                  <Input type="date" value={form.endDate} onChange={e=>setForm({...form, endDate:e.target.value})}/>
                </HStack>
              </FormRow>
              <FormRow label="Per diem">
                <NumberInput value={form.perDiem} onChange={(_,num)=>setForm({...form, perDiem: num||0})} min={0}>
                  <NumberInputField/>
                </NumberInput>
              </FormRow>
              <FormRow label="Need advance?">
                <Switch isChecked={form.needAdvance} onChange={e=>setForm({...form, needAdvance:e.target.checked})}/>
              </FormRow>
              <FormRow label="Notes"><Textarea placeholder="Purpose, itinerary, etc."/></FormRow>
            </VStack>
          </DrawerBody>
          <DrawerFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>Cancel</Button>
            <Button colorScheme="purple" onClick={create}>Create draft</Button>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    </AppShell>
  );
}

function RequestsTable({items, onSubmit, compact}){
  const border = useColorModeValue("gray.200","gray.700");
  return (
    <Box bg={useColorModeValue("white","gray.800")} borderRadius="2xl" borderWidth={1} overflow="hidden">
      <Table size={compact?"sm":"md"}>
        <Thead>
          <Tr>
            <Th>Title</Th>
            <Th>Project</Th>
            <Th>Dates</Th>
            <Th>Status</Th>
            <Th isNumeric>Estimate</Th>
            <Th></Th>
          </Tr>
        </Thead>
        <Tbody>
          {items.map(r=> (
            <Tr key={r.id} _hover={{bg: useColorModeValue("gray.50","gray.700")}}>
              <Td>{r.title}</Td>
              <Td>{projects.find(p=>p.id===r.projectId)?.name}</Td>
              <Td><Text fontSize="sm" color="gray.500">{r.startDate} → {r.endDate}</Text></Td>
              <Td>
                <HStack>
                  {r.status==="Approved" && <Badge colorScheme="green" leftIcon={<CheckCircle2 size={12}/>}>Approved</Badge>}
                  {r.status==="Submitted" && <Badge colorScheme="purple">Submitted</Badge>}
                  {r.status==="Draft" && <Badge>Draft</Badge>}
                </HStack>
              </Td>
              <Td isNumeric>{(r.estimate||0).toFixed(2)} EUR</Td>
              <Td textAlign="right">
                {onSubmit && r.status==="Draft" && <Button size="sm" variant="outline" onClick={()=>onSubmit(r.id)}>Submit</Button>}
                {onSubmit && r.status==="Submitted" && <Button size="sm" isDisabled>Awaiting approval</Button>}
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Box>
  );
}

function ExpensesPage(){
  const [selectedRequest, setSelectedRequest] = useState(101);
  const [items, setItems] = useState([]);
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({ categoryId: 3, amount: "", currency: "EUR", spendAt: "", merchant: "", note: "" });
  const toast = useToast();

  useEffect(()=>{ api.getExpensesByRequest(selectedRequest).then(setItems); },[selectedRequest]);

  async function extract(){
    if(!file){ toast({ title:"Attach a file first", status:"warning"}); return; }
    const res = await api.ocrExtract(file);
    setForm(f=>({ ...f, amount: String(res.amount), spendAt: res.date, merchant: res.merchant }));
    toast({ title:"OCR extracted (demo)", description:`Confidence ${res.confidence}`, status:"info"});
  }

  async function addExpense(){
    const payload = { requestId: selectedRequest, employeeId: 1, fileName: file?.name || "receipt.jpg", ...form, amount: Number(form.amount), categoryId: Number(form.categoryId) };
    const created = await api.createExpense(payload);
    setItems(prev=>[created, ...prev]);
    setForm({ categoryId: 3, amount: "", currency: "EUR", spendAt: "", merchant: "", note: "" });
    setFile(null);
    toast({ title:"Expense added", status:"success"});
  }

  return (
    <AppShell>
      <Heading size="md" mb={4}>Expenses</Heading>
      <VStack align="stretch" spacing={6}>
        <Box bg={useColorModeValue("white","gray.800")} borderRadius="2xl" p={4} borderWidth={1}>
          <HStack mb={3}>
            <Text fontWeight="semibold">Request</Text>
            <Select value={selectedRequest} onChange={e=>setSelectedRequest(Number(e.target.value))} maxW={300}>
              {store.travelRequests.map(r=> <option key={r.id} value={r.id}>{r.title}</option>)}
            </Select>
          </HStack>
          <SimpleGrid columns={{base:1, md:2}} spacing={4}>
            <FormRow label="Category">
              <Select value={form.categoryId} onChange={e=>setForm({...form, categoryId: Number(e.target.value)})}>
                {categories.map(c=> <option key={c.id} value={c.id}>{c.name}</option>)}
              </Select>
            </FormRow>
            <FormRow label="Amount">
              <Input type="number" step="0.01" value={form.amount} onChange={e=>setForm({...form, amount:e.target.value})}/>
            </FormRow>
            <FormRow label="Currency">
              <Select value={form.currency} onChange={e=>setForm({...form, currency:e.target.value})}>
                <option>EUR</option>
                <option>USD</option>
                <option>GEL</option>
              </Select>
            </FormRow>
            <FormRow label="Date">
              <Input type="date" value={form.spendAt} onChange={e=>setForm({...form, spendAt:e.target.value})}/>
            </FormRow>
            <FormRow label="Merchant">
              <Input value={form.merchant} onChange={e=>setForm({...form, merchant:e.target.value})}/>
            </FormRow>
            <FormRow label="Note">
              <Input value={form.note} onChange={e=>setForm({...form, note:e.target.value})}/>
            </FormRow>
          </SimpleGrid>
          <HStack mt={3}>
            <Input type="file" onChange={e=>setFile(e.target.files?.[0]||null)} accept="image/*,application/pdf"/>
            <Button leftIcon={<Camera size={16}/>} variant="outline" onClick={extract}>Extract (demo)</Button>
            <Spacer/>
            <Button colorScheme="purple" leftIcon={<Upload size={16}/>} onClick={addExpense}>Add expense</Button>
          </HStack>
        </Box>

        <Box bg={useColorModeValue("white","gray.800")} borderRadius="2xl" borderWidth={1}>
          <Table>
            <Thead>
              <Tr>
                <Th>Date</Th>
                <Th>Category</Th>
                <Th>Merchant</Th>
                <Th isNumeric>Amount</Th>
                <Th>File</Th>
                <Th>OCR</Th>
              </Tr>
            </Thead>
            <Tbody>
              {items.map(e=> (
                <Tr key={e.id}>
                  <Td>{e.spendAt}</Td>
                  <Td>{categories.find(c=>c.id===e.categoryId)?.name}</Td>
                  <Td>{e.merchant}</Td>
                  <Td isNumeric>{e.amount.toFixed(2)} {e.currency}</Td>
                  <Td><Tag size="sm">{e.fileName}</Tag></Td>
                  <Td>{e.verified? <Badge colorScheme="green">Verified</Badge> : <Badge colorScheme="yellow">Pending</Badge>}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Box>
      </VStack>
    </AppShell>
  );
}

function ReportsPage(){
  const [reports, setReports] = useState([]);
  const toast = useToast();
  useEffect(()=>{ api.getReports().then(setReports); },[]);

  async function createFrom(reqId){
    const rep = await api.createReportFromRequest(reqId);
    setReports(prev=>[rep, ...prev]);
    toast({ title:"Report created from request", status:"success"});
  }

  return (
    <AppShell>
      <Heading size="md" mb={4}>Reports</Heading>
      <Tabs variant='enclosed'>
        <TabList>
          <Tab>My reports</Tab>
          <Tab>Create from request</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Box bg={useColorModeValue("white","gray.800")} borderRadius="2xl" borderWidth={1}>
              <Table>
                <Thead>
                  <Tr>
                    <Th>ID</Th>
                    <Th>Request</Th>
                    <Th>Status</Th>
                    <Th isNumeric>Total</Th>
                    <Th>Submitted</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {reports.map(r=> (
                    <Tr key={r.id}>
                      <Td>{r.id}</Td>
                      <Td>{store.travelRequests.find(t=>t.id===r.requestId)?.title}</Td>
                      <Td><Badge colorScheme={r.status==="Closed"?"green":"purple"}>{r.status}</Badge></Td>
                      <Td isNumeric>{r.total.toFixed(2)} EUR</Td>
                      <Td>{r.submittedAt}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </Box>
          </TabPanel>
          <TabPanel>
            <VStack align="stretch" spacing={3}>
              <Text>Select a request to generate report (demo):</Text>
              <SimpleGrid columns={{base:1, md:2}} spacing={3}>
                {store.travelRequests.map(r=> (
                  <Box key={r.id} p={4} borderWidth={1} borderRadius="xl" bg={useColorModeValue("white","gray.800")}> 
                    <Heading size="sm" mb={2}>{r.title}</Heading>
                    <Text fontSize="sm" color="gray.500">{r.startDate} → {r.endDate}</Text>
                    <HStack mt={2}>
                      <Button size="sm" leftIcon={<FileText size={14}/>} onClick={()=>createFrom(r.id)}>Create report</Button>
                    </HStack>
                  </Box>
                ))}
              </SimpleGrid>
            </VStack>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </AppShell>
  );
}

function BudgetsPage(){
  const [items, setItems] = useState([]);
  useEffect(()=>{ api.getBudgets().then(setItems); },[]);
  return (
    <AppShell>
      <Heading size="md" mb={4}>Budgets</Heading>
      <SimpleGrid columns={{base:1, md:2}} spacing={4}>
        {items.map(b=> <BudgetCard key={b.id} budget={b}/>) }
      </SimpleGrid>
    </AppShell>
  );
}

function AdminPage(){
  const [enabledOCR, setEnabledOCR] = useState(true);
  const [autoApproveUnder, setAutoApproveUnder] = useState(0);
  const toast = useToast();
  return (
    <AppShell>
      <Heading size="md" mb={6}>Admin (Demo settings)</Heading>
      <VStack align="stretch" spacing={4}>
        <Box p={4} borderRadius="2xl" borderWidth={1} bg={useColorModeValue("white","gray.800")}> 
          <Heading size="sm" mb={3}>Feature flags</Heading>
          <HStack>
            <Switch isChecked={enabledOCR} onChange={e=>setEnabledOCR(e.target.checked)}/> 
            <Text>Enable OCR button on Expenses page</Text>
          </HStack>
        </Box>
        <Box p={4} borderRadius="2xl" borderWidth={1} bg={useColorModeValue("white","gray.800")}> 
          <Heading size="sm" mb={3}>Policies</Heading>
          <HStack maxW={400}>
            <Text flexShrink={0}>Auto-approve if estimate under (EUR):</Text>
            <NumberInput value={autoApproveUnder} onChange={(_,n)=>setAutoApproveUnder(n||0)} min={0}><NumberInputField/></NumberInput>
            <Button onClick={()=>toast({title:"Saved (demo)", status:"success"})}>Save</Button>
          </HStack>
        </Box>
      </VStack>
    </AppShell>
  );
}

function FormRow({label, children}){
  return (
    <Box>
      <Text fontSize="sm" mb={1} color="gray.500">{label}</Text>
      {children}
    </Box>
  );
}

// ---- ROOT APP ----
function App(){
  return (
    <ChakraProvider theme={theme}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginPage/>} />
          <Route path="/dashboard" element={<DashboardPage/>} />
          <Route path="/travel" element={<TravelRequestsPage/>} />
          <Route path="/expenses" element={<ExpensesPage/>} />
          <Route path="/reports" element={<ReportsPage/>} />
          <Route path="/budgets" element={<BudgetsPage/>} />
          <Route path="/admin" element={<AdminPage/>} />
        </Routes>
      </BrowserRouter>
    </ChakraProvider>
  );
}

export default App;
