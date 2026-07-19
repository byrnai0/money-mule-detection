import {motion} from "framer-motion";
export function Title({eyebrow,title,desc}:{eyebrow:string,title:string,desc:string}){return <div className="title"><span>{eyebrow}</span><h1>{title}</h1><p>{desc}</p></div>}
export function Card({children,className=""}:{children:React.ReactNode,className?:string}){return <motion.div initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} className={"card "+className}>{children}</motion.div>}
export function Risk({v}:{v:number}){return <span className={"risk "+(v>=90?"critical":v>=75?"high":"medium")}>{v.toFixed(1)}</span>}